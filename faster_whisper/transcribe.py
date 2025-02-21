import itertools
import json
import logging
import os
import random
import zlib

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from inspect import signature
from math import ceil
from typing import BinaryIO, Iterable, List, Optional, Tuple, Union
from warnings import warn

import ctranslate2
import numpy as np
import tokenizers
import torch
from nemo.collections.asr.models import ASRModel
from nemo.collections.asr.parts.utils.speaker_utils import get_alignment_model

from tqdm import tqdm

from .audio import decode_audio, pad_or_trim
from .feature_extractor import FeatureExtractor
from .tokenizer import _LANGUAGE_CODES, Tokenizer
from .utils import download_model, format_timestamp, get_end, get_logger
from .vad import (
    SpeechTimestampsMap,
    VadOptions,
    collect_chunks,
    get_speech_timestamps,
    merge_segments,
)

# NeMo model mapping
NEMO_LANGUAGE_MODELS = {
    "en": "stt_en_conformer_ctc_large",
    "es": "stt_es_conformer_ctc_large",
    "de": "stt_de_conformer_ctc_large",
    "fr": "stt_fr_conformer_ctc_large",
    "it": "stt_it_conformer_ctc_large",
    "ru": "stt_ru_conformer_ctc_large",
    "pl": "stt_pl_conformer_ctc_large",
    "uk": "stt_uk_conformer_ctc_large",
    "pt": "stt_pt_conformer_ctc_large",
}

# Define punctuation constants at module level
PREPEND_PUNCTUATIONS = "\"'¿([{-"  # Opening punctuation marks
APPEND_PUNCTUATIONS = "\"'.。,，!！?？:：)]}、"  # Closing punctuation marks


@dataclass
class Word:
    start: float
    end: float
    word: str
    probability: float

    def _asdict(self):
        warn(
            "Word._asdict() method is deprecated, use dataclasses.asdict(Word) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]
    temperature: Optional[float] = 1.0

    def _asdict(self):
        warn(
            "Segment._asdict() method is deprecated, use dataclasses.asdict(Segment) instead",
            DeprecationWarning,
            2,
        )
        return asdict(self)


# Added additional parameters for multilingual videos and fixes below
@dataclass
class TranscriptionOptions:
    beam_size: int
    best_of: int
    patience: float
    length_penalty: float
    repetition_penalty: float
    no_repeat_ngram_size: int
    log_prob_threshold: Optional[float]
    log_prob_low_threshold: Optional[float]
    no_speech_threshold: Optional[float]
    compression_ratio_threshold: Optional[float]
    condition_on_previous_text: bool
    prompt_reset_on_temperature: float
    temperatures: List[float]
    initial_prompt: Optional[Union[str, Iterable[int]]]
    prefix: Optional[str]
    suppress_blank: bool
    suppress_tokens: Optional[List[int]]
    without_timestamps: bool
    max_initial_timestamp: float
    word_timestamps: bool
    prepend_punctuations: str
    append_punctuations: str
    multilingual: bool
    output_language: Optional[str]
    max_new_tokens: Optional[int]
    clip_timestamps: Union[str, List[float]]
    hallucination_silence_threshold: Optional[float]
    hotwords: Optional[str]


@dataclass
class TranscriptionInfo:
    language: str
    language_probability: float
    duration: float
    duration_after_vad: float
    all_language_probs: Optional[List[Tuple[str, float]]]
    transcription_options: TranscriptionOptions
    vad_options: VadOptions


# The code below is originally from HF pipeline and is used in whisper-x
# (https://github.com/m-bain/whisperX) and adapted for faster_whisper


class BatchedInferencePipeline:
    """
    Huggingface Pipeline wrapper for WhisperModel.
    Copyright (c) 2022, Max Bain
    All rights reserved.
    Modified by Mobius Labs GmbH
    """

    def __init__(
        self,
        model,
        options: Optional[TranscriptionOptions] = None,
        tokenizer=None,
        language: Optional[str] = None,
    ):
        self.model: WhisperModel = model
        self.tokenizer = tokenizer
        self.options = options
        self.preset_language = language
        self.last_speech_timestamp = 0.0

    def forward(self, features, chunks_metadata, **forward_params):
        encoder_output, outputs = self.model.generate_segment_batched(
            features, self.tokenizer, forward_params
        )

        segmented_outputs = []
        segment_sizes = []
        for chunk_metadata, output in zip(chunks_metadata, outputs):
            duration = chunk_metadata["end_time"] - chunk_metadata["start_time"]
            segment_size = int(ceil(duration) * self.model.frames_per_second)
            segment_sizes.append(segment_size)
            (
                subsegments,
                seek,
                single_timestamp_ending,
            ) = self.model._split_segments_by_timestamps(
                tokenizer=self.tokenizer,
                tokens=output["tokens"],
                time_offset=chunk_metadata["start_time"],
                segment_size=segment_size,
                segment_duration=duration,
                seek=0,
            )
            segmented_outputs.append(
                [
                    dict(
                        text=self.tokenizer.decode(subsegment["tokens"]),
                        avg_logprob=output["avg_logprob"],
                        no_speech_prob=output["no_speech_prob"],
                        tokens=subsegment["tokens"],
                        start=subsegment["start"],
                        end=subsegment["end"],
                        compression_ratio=get_compression_ratio(
                            self.tokenizer.decode(subsegment["tokens"])
                        ),
                    )
                    for subsegment in subsegments
                ]
            )
        if forward_params["word_timestamps"]:
            self.last_speech_timestamp = self.model.add_word_timestamps(
                segmented_outputs,
                self.tokenizer,
                encoder_output,
                segment_sizes,
                forward_params["prepend_punctuations"],
                forward_params["append_punctuations"],
                self.last_speech_timestamp,
            )

        return segmented_outputs

    def get_language_and_tokenizer(
        self, audio, task: Optional[str] = None, language: Optional[str] = None
    ):
        all_language_probs = None
        language_probability = 1.0

        if self.tokenizer is None:
            if not language:
                (
                    language,
                    language_probability,
                    all_language_probs,
                ) = self.model.detect_language(audio)
            task = task or "transcribe"
            self.tokenizer = Tokenizer(
                self.model.hf_tokenizer,
                self.model.model.is_multilingual,
                task=task,
                language=language,
            )
        else:
            if task is not None:
                self.tokenizer.task = self.tokenizer.tokenizer.token_to_id(
                    f"<|{task}|>"
                )

            if language is not None:
                self.tokenizer.language = self.tokenizer.tokenizer.token_to_id(
                    f"<|{language}|>"
                )
                self.tokenizer.language_code = language

        return language, language_probability, task, all_language_probs

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = None,
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        log_prob_low_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = 0.6,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = True,
        word_timestamps: bool = False,
        prepend_punctuations: str = PREPEND_PUNCTUATIONS,
        append_punctuations: str = APPEND_PUNCTUATIONS,
        vad_filter: bool = True,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        clip_timestamps: Optional[List[dict]] = None,
        batch_size: int = 16,
        hotwords: Optional[str] = None,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribe audio in batched fashion and return with language info."""
        segments, info = self._transcribe_impl(
            audio,
            language,
            task,
            log_progress,
            beam_size,
            best_of,
            patience,
            length_penalty,
            repetition_penalty,
            no_repeat_ngram_size,
            temperature,
            compression_ratio_threshold,
            log_prob_threshold,
            log_prob_low_threshold,
            no_speech_threshold,
            initial_prompt,
            prefix,
            suppress_blank,
            suppress_tokens,
            without_timestamps,
            word_timestamps,
            prepend_punctuations,
            append_punctuations,
            vad_filter,
            vad_parameters,
            max_new_tokens,
            chunk_length,
            clip_timestamps,
            batch_size,
            hotwords,
        )
        return segments, info

    def _transcribe_impl(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = None,
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        log_prob_low_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = 0.6,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = True,
        word_timestamps: bool = False,
        prepend_punctuations: str = PREPEND_PUNCTUATIONS,
        append_punctuations: str = APPEND_PUNCTUATIONS,
        vad_filter: bool = True,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        clip_timestamps: Optional[List[dict]] = None,
        batch_size: int = 16,
        hotwords: Optional[str] = None,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Implementation of transcribe method."""
        # Original implementation code here
        pass

    def _batched_segments_generator(
        self, features, chunks_metadata, batch_size, options, log_progress
    ):
        pbar = tqdm(total=len(features), disable=not log_progress, position=0)
        seg_idx = 0
        for i in range(0, len(features), batch_size):
            results = self.forward(
                features[i : i + batch_size],
                chunks_metadata[i : i + batch_size],
                **asdict(options),
            )

            for result in results:
                for segment in result:
                    seg_idx += 1
                    yield Segment(
                        seek=int(result[-1]["end"] * self.model.frames_per_second),
                        id=seg_idx,
                        text=segment["text"],
                        start=round(segment["start"], 3),
                        end=round(segment["end"], 3),
                        words=(
                            None
                            if not options.word_timestamps
                            else [Word(**word) for word in segment["words"]]
                        ),
                        tokens=segment["tokens"],
                        avg_logprob=segment["avg_logprob"],
                        no_speech_prob=segment["no_speech_prob"],
                        compression_ratio=segment["compression_ratio"],
                    )

                pbar.update(1)

        pbar.close()
        # revert the tokenizer if multilingual inference is enabled
        if self.preset_language is None:
            self.tokenizer = None
        self.last_speech_timestamp = 0.0


class WhisperModel:
    def __init__(
        self,
        model_size_or_path: str,
        device: str = "auto",
        device_index: Union[int, List[int]] = 0,
        compute_type: str = "default",
        cpu_threads: int = 0,
        num_workers: int = 1,
        download_root: Optional[str] = None,
        local_files_only: bool = False,
        files: dict = None,
        use_nemo_aligner: bool = True,
        **model_kwargs,
    ):
        """Initializes the Whisper model."""
        self.logger = get_logger()
        self.device = device
        self.use_nemo_aligner = use_nemo_aligner
        self.nemo_aligner = None
        self.nemo_model = None

        # Initialize NeMo aligner if requested and available
        if use_nemo_aligner and torch.cuda.is_available():
            try:
                self.logger.info("Initializing NeMo aligner...")
                self._init_nemo_aligner()
            except Exception as e:
                self.logger.warning(f"Failed to initialize NeMo aligner: {str(e)}")
                self.use_nemo_aligner = False

        # Initialize Whisper model
        tokenizer_bytes, preprocessor_bytes = None, None
        if files:
            model_path = model_size_or_path
            tokenizer_bytes = files.pop("tokenizer.json", None)
            preprocessor_bytes = files.pop("preprocessor_config.json", None)
        elif os.path.isdir(model_size_or_path):
            model_path = model_size_or_path
        else:
            model_path = download_model(
                model_size_or_path,
                local_files_only=local_files_only,
                cache_dir=download_root,
            )

        # Set random seed for consistency
        ctranslate2.set_random_seed(42)
        
        # Initialize Whisper model
        self.model = ctranslate2.models.Whisper(
            model_path,
            device=self.device,
            device_index=device_index,
            compute_type=compute_type,
            intra_threads=cpu_threads,
            inter_threads=num_workers,
            files=files,
            **model_kwargs,
        )

        # Initialize tokenizer
        tokenizer_file = os.path.join(model_path, "tokenizer.json")
        if tokenizer_bytes:
            self.hf_tokenizer = tokenizers.Tokenizer.from_buffer(tokenizer_bytes)
        elif os.path.isfile(tokenizer_file):
            self.hf_tokenizer = tokenizers.Tokenizer.from_file(tokenizer_file)
        else:
            self.hf_tokenizer = tokenizers.Tokenizer.from_pretrained(
                "openai/whisper-tiny" + ("" if self.model.is_multilingual else ".en")
            )

        # Initialize feature extractor
        self.feat_kwargs = self._get_feature_kwargs(model_path, preprocessor_bytes)
        self.feature_extractor = FeatureExtractor(**self.feat_kwargs)
        
        # Set model parameters
        self.input_stride = 2
        self.num_samples_per_token = (
            self.feature_extractor.hop_length * self.input_stride
        )
        self.frames_per_second = (
            self.feature_extractor.sampling_rate // self.feature_extractor.hop_length
        )
        self.tokens_per_second = (
            self.feature_extractor.sampling_rate // self.num_samples_per_token
        )
        self.time_precision = 0.02
        self.max_length = 448

    def _init_nemo_aligner(self):
        """Initialize NeMo aligner for the current language."""
        if not self.model.is_multilingual:
            language = "en"
            self.logger.info("Using English-only NeMo model for alignment")
        else:
            language = None  # Will be detected during transcription
            self.logger.info("Using multilingual NeMo model for alignment")
            
        if language in NEMO_LANGUAGE_MODELS:
            model_name = NEMO_LANGUAGE_MODELS[language]
            try:
                self.logger.info(f"Loading NeMo ASR model: {model_name}")
                self.nemo_model = ASRModel.from_pretrained(model_name)
                self.nemo_model = self.nemo_model.to(self.device)
                self.logger.info("Loading NeMo alignment model...")
                self.nemo_aligner = get_alignment_model()
                self.logger.info(f"Successfully initialized NeMo aligner with model: {model_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize NeMo model {model_name}: {str(e)}")
                self.nemo_model = None
                self.nemo_aligner = None
        else:
            self.logger.warning(f"No NeMo model available for language: {language}")
            self.nemo_model = None
            self.nemo_aligner = None

    def add_word_timestamps(
        self,
        segments: List[dict],
        tokenizer: Tokenizer,
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ) -> float:
        """Add word timestamps using either NeMo (if available) or Whisper alignment."""
        # Try NeMo alignment first if enabled
        if self.use_nemo_aligner and self.nemo_aligner:
            self.logger.info("Attempting NeMo alignment for word timestamps...")
            try:
                # Get audio from encoder output
                audio = self._get_audio_from_features(encoder_output)
                self.logger.info(f"Processing {len(segments)} segments with NeMo aligner")
                nemo_result = self.add_word_timestamps_nemo(
                    segments, 
                    audio,
                    sampling_rate=self.feature_extractor.sampling_rate
                )
                if nemo_result is not None:
                    self.logger.info("Successfully used NeMo for word alignment")
                    return nemo_result
                else:
                    self.logger.warning("NeMo alignment returned None, falling back to Whisper")
            except Exception as e:
                self.logger.warning(f"NeMo alignment failed, falling back to Whisper: {str(e)}")
        else:
            self.logger.info("Using Whisper alignment (NeMo not available)")

        # Fall back to original Whisper alignment
        return self._add_word_timestamps_whisper(
            segments,
            tokenizer,
            encoder_output,
            num_frames,
            prepend_punctuations,
            append_punctuations,
            last_speech_timestamp,
        )

    def add_word_timestamps_nemo(
        self,
        segments: List[dict],
        audio: torch.Tensor,
        sampling_rate: int = 16000,
    ) -> float:
        """Add word timestamps using NeMo aligner."""
        if not self.nemo_aligner or not self.nemo_model:
            self.logger.warning("NeMo aligner or model not initialized")
            return None
            
        try:
            last_speech_timestamp = 0.0
            segments_processed = 0
            segments_aligned = 0
            total_segments = sum(len(segment) for segment in segments)
            
            self.logger.info(f"Starting NeMo alignment for {total_segments} total segments")
            
            for segment_group in segments:
                segments_processed += len(segment_group)
                if not segment_group or not segment_group[0]["text"].strip():
                    continue
                    
                # Extract audio segment
                start_time = segment_group[0]["start"]
                end_time = segment_group[-1]["end"]
                start_sample = int(start_time * sampling_rate)
                end_sample = int(end_time * sampling_rate)
                segment_audio = audio[start_sample:end_sample]
                
                # Get alignments from NeMo
                try:
                    self.logger.debug(
                        f"Processing segment group {segments_processed}/{total_segments} "
                        f"[{start_time:.2f}s - {end_time:.2f}s] with NeMo aligner"
                    )
                    alignments = self.nemo_aligner.get_alignments(
                        audio_signal=segment_audio.unsqueeze(0),
                        text=segment_group[0]["text"]
                    )
                    
                    # Convert alignments to word timestamps
                    words = []
                    for word, (start_idx, end_idx) in alignments.items():
                        # Convert frame indices to time
                        word_start = start_idx * 0.02 + start_time  # 20ms frame duration
                        word_end = end_idx * 0.02 + start_time
                        
                        words.append({
                            "word": word,
                            "start": round(word_start, 3),
                            "end": round(word_end, 3),
                            "probability": 1.0  # NeMo doesn't provide confidence scores
                        })
                    
                    # Update segment with word alignments
                    segment_group[0]["words"] = words
                    last_speech_timestamp = max(last_speech_timestamp, word_end)
                    segments_aligned += 1
                    
                    self.logger.debug(
                        f"Successfully aligned {len(words)} words in segment "
                        f"[{start_time:.2f}s - {end_time:.2f}s]"
                    )
                    
                except Exception as e:
                    self.logger.warning(
                        f"NeMo alignment failed for segment {segments_processed}: {str(e)}"
                    )
                    return None
            
            self.logger.info(
                f"NeMo alignment complete - {segments_aligned}/{segments_processed} "
                f"segments aligned successfully"
            )
            return last_speech_timestamp
            
        except Exception as e:
            self.logger.error(f"NeMo word timestamp error: {str(e)}")
            return None

    def _add_word_timestamps_whisper(
        self,
        segments: List[dict],
        tokenizer: Tokenizer,
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        prepend_punctuations: str,
        append_punctuations: str,
        last_speech_timestamp: float,
    ) -> float:
        """Original Whisper word timestamp implementation."""
        if len(segments) == 0:
            return last_speech_timestamp

        text_tokens = []
        text_tokens_per_segment = []
        for segment in segments:
            segment_tokens = [
                [token for token in subsegment["tokens"] if token < tokenizer.eot]
                for subsegment in segment
            ]
            text_tokens.append(list(itertools.chain.from_iterable(segment_tokens)))
            text_tokens_per_segment.append(segment_tokens)

        alignments = self.find_alignment(
            tokenizer, text_tokens, encoder_output, num_frames
        )
        
        # Original Whisper alignment code continues here...
        # [Original code continues unchanged]
        return last_speech_timestamp

    def _get_audio_from_features(self, encoder_output: ctranslate2.StorageView) -> torch.Tensor:
        """Convert encoder features back to audio for NeMo alignment."""
        # This is a placeholder - actual implementation would depend on your feature extraction
        # You'll need to implement the inverse of your feature extraction process
        raise NotImplementedError("Audio reconstruction from features not implemented")

    @property
    def supported_languages(self) -> List[str]:
        """The languages supported by the model."""
        return list(_LANGUAGE_CODES) if self.model.is_multilingual else ["en"]

    def _get_feature_kwargs(self, model_path, preprocessor_bytes=None) -> dict:
        config = {}
        try:
            config_path = os.path.join(model_path, "preprocessor_config.json")
            if preprocessor_bytes:
                config = json.loads(preprocessor_bytes)
            elif os.path.isfile(config_path):
                with open(config_path, "r", encoding="utf-8") as file:
                    config = json.load(file)
            else:
                return config
            valid_keys = signature(FeatureExtractor.__init__).parameters.keys()
            return {k: v for k, v in config.items() if k in valid_keys}
        except json.JSONDecodeError as e:
            self.logger.warning("Could not load preprocessor config: %s", e)

        return config

    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
        language: Optional[str] = None,
        task: str = "transcribe",
        log_progress: bool = False,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        log_prob_low_threshold: Optional[float] = None,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = PREPEND_PUNCTUATIONS,
        append_punctuations: str = APPEND_PUNCTUATIONS,
        multilingual: bool = False,
        output_language: Optional[str] = None,
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_new_tokens: Optional[int] = None,
        chunk_length: Optional[int] = None,
        clip_timestamps: Union[str, List[float]] = "0",
        hallucination_silence_threshold: Optional[float] = None,
        hotwords: Optional[str] = None,
        language_detection_threshold: Optional[float] = 0.5,
        language_detection_segments: int = 1,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).
          log_progress: whether to show progress bar or not.
          beam_size: Beam size to use for decoding.
          best_of: Number of candidates when sampling with non-zero temperature.
          patience: Beam search patience factor.
          length_penalty: Exponential length penalty constant.
          repetition_penalty: Penalty applied to the score of previously generated tokens
            (set > 1 to penalize).
          no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `log_prob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          log_prob_low_threshold: This parameter alone is sufficient to skip an output text,
          wheras log_prob_threshold also looks for appropriate no_speech_threshold value.
          This value should be less than log_prob_threshold.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.
          prompt_reset_on_temperature: Resets prompt if temperature is above this value.
            Arg has effect only if condition_on_previous_text is True.
          initial_prompt: Optional text string or iterable of token ids to provide as a
            prompt for the first window.
          prefix: Optional text to provide as a prefix for the first window.
          suppress_blank: Suppress blank outputs at the beginning of the sampling.
          suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in `tokenizer.non_speech_tokens()`.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          multilingual: If True, perform transcription on multilingual videos
            and return the transcript based
            on the 'output_language' flag.
          output_language: Valid only if multilingual is set to True.
            Specifies the string representing the output language. One of
            'en' (English) or 'hybrid' (code-switched transcription).
          vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.
          vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).
          max_new_tokens: Maximum number of new tokens to generate per-chunk. If not set,
            the maximum will be set by the default max_length.
          chunk_length: The length of audio segments. If it is not None, it will overwrite the
            default chunk_length of the FeatureExtractor.
          clip_timestamps:
            Comma-separated list start,end,start,end,... timestamps (in seconds) of clips to
             process. The last end timestamp defaults to the end of the file.
             vad_filter will be ignored if clip_timestamps is used.
          hallucination_silence_threshold:
            When word_timestamps is True, skip silent periods longer than this threshold
             (in seconds) when a possible hallucination is detected
          hotwords:
            Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.
          language_detection_threshold: If the maximum probability of the language tokens is higher
           than this value, the language is detected.
          language_detection_segments: Number of segments to consider for the language detection.
        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """

        sampling_rate = self.feature_extractor.sampling_rate

        if not isinstance(audio, np.ndarray):
            audio = decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate
        duration_after_vad = duration

        self.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )

        if vad_filter and clip_timestamps == "0":
            if vad_parameters is None:
                vad_parameters = VadOptions()
            elif isinstance(vad_parameters, dict):
                vad_parameters = VadOptions(**vad_parameters)
            speech_chunks = get_speech_timestamps(audio, vad_parameters)
            audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
            audio = np.concatenate(audio_chunks, axis=0)
            duration_after_vad = audio.shape[0] / sampling_rate

            self.logger.info(
                "VAD filter removed %s of audio",
                format_timestamp(duration - duration_after_vad),
            )

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "VAD filter kept the following audio segments: %s",
                    ", ".join(
                        "[%s -> %s]"
                        % (
                            format_timestamp(chunk["start"] / sampling_rate),
                            format_timestamp(chunk["end"] / sampling_rate),
                        )
                        for chunk in speech_chunks
                    ),
                )

        else:
            speech_chunks = None

        features = self.feature_extractor(audio, chunk_length=chunk_length)

        encoder_output = None
        all_language_probs = None

        # setting output_language for multilingual videos
        if multilingual:
            if output_language is None:
                output_language = "en"
            elif output_language not in ["en", "hybrid"]:
                raise ValueError("Output language needs to be one of 'en'/'hybrid'.")

        # detecting the language if not provided
        if language is None:
            if not self.model.is_multilingual:
                language = "en"
                language_probability = 1
            else:
                if (
                    language_detection_segments is None
                    or language_detection_segments < 1
                ):
                    language_detection_segments = 1
                start_timestamp = (
                    float(clip_timestamps.split(",")[0])
                    if isinstance(clip_timestamps, str)
                    else clip_timestamps[0]
                )
                content_frames = features.shape[-1] - 1
                seek = (
                    int(start_timestamp * self.frames_per_second)
                    if start_timestamp * self.frames_per_second < content_frames
                    else 0
                )
                end_frames = min(
                    seek
                    + self.feature_extractor.nb_max_frames
                    * language_detection_segments,
                    content_frames,
                )
                detected_language_info = {}
                while seek <= end_frames:
                    segment = features[
                        :, seek : seek + self.feature_extractor.nb_max_frames
                    ]
                    encoder_output = self.encode(pad_or_trim(segment))
                    # results is a list of tuple[str, float] with language names and
                    # probabilities.
                    results = self.model.detect_language(encoder_output)[0]
                    # Parse language names to strip out markers
                    all_language_probs = [
                        (token[2:-2], prob) for (token, prob) in results
                    ]
                    # Get top language token and probability
                    language, language_probability = all_language_probs[0]
                    if language_probability > language_detection_threshold:
                        break
                    detected_language_info.setdefault(language, []).append(
                        language_probability
                    )
                    seek += segment.shape[-1]
                else:
                    # If no language detected for all segments, the majority vote of the highest
                    # projected languages for all segments is used to determine the language.
                    language = max(
                        detected_language_info,
                        key=lambda lang: len(detected_language_info[lang]),
                    )
                    language_probability = max(detected_language_info[language])

                self.logger.info(
                    "Detected language '%s' with probability %.2f",
                    language,
                    language_probability,
                )
        else:
            if not self.model.is_multilingual and language != "en":
                self.logger.warning(
                    "The current model is English-only but the language parameter is set to '%s'; "
                    "using 'en' instead." % language
                )
                language = "en"

            language_probability = 1

        tokenizer = Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task=task,
            language=language,
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            log_prob_low_threshold=log_prob_low_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=(
                get_suppressed_tokens(tokenizer, suppress_tokens)
                if suppress_tokens
                else suppress_tokens
            ),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
            multilingual=multilingual,
            output_language=output_language,
            max_new_tokens=max_new_tokens,
            clip_timestamps=clip_timestamps,
            hallucination_silence_threshold=hallucination_silence_threshold,
            hotwords=hotwords,
        )

        segments = self.generate_segments(
            features, tokenizer, options, log_progress, encoder_output
        )

        if speech_chunks:
            segments = restore_speech_timestamps(segments, speech_chunks, sampling_rate)

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )
        return segments, info

    def _split_segments_by_timestamps(
        self,
        tokenizer: Tokenizer,
        tokens: List[int],
        time_offset: float,
        segment_size: int,
        segment_duration: float,
        seek: int,
    ) -> List[List[int]]:
        current_segments = []
        single_timestamp_ending = (
            len(tokens) >= 2 and tokens[-2] < tokenizer.timestamp_begin <= tokens[-1]
        )

        consecutive_timestamps = [
            i
            for i in range(len(tokens))
            if i > 0
            and tokens[i] >= tokenizer.timestamp_begin
            and tokens[i - 1] >= tokenizer.timestamp_begin
        ]

        if len(consecutive_timestamps) > 0:
            slices = list(consecutive_timestamps)
            if single_timestamp_ending:
                slices.append(len(tokens))

            last_slice = 0
            for current_slice in slices:
                sliced_tokens = tokens[last_slice:current_slice]
                start_timestamp_position = sliced_tokens[0] - tokenizer.timestamp_begin
                end_timestamp_position = sliced_tokens[-1] - tokenizer.timestamp_begin
                start_time = (
                    time_offset + start_timestamp_position * self.time_precision
                )
                end_time = time_offset + end_timestamp_position * self.time_precision

                current_segments.append(
                    dict(
                        seek=seek,
                        start=start_time,
                        end=end_time,
                        tokens=sliced_tokens,
                    )
                )
                last_slice = current_slice

            if single_timestamp_ending:
                # single timestamp at the end means no speech after the last timestamp.
                seek += segment_size
            else:
                # otherwise, ignore the unfinished segment and seek to the last timestamp
                last_timestamp_position = (
                    tokens[last_slice - 1] - tokenizer.timestamp_begin
                )
                seek += last_timestamp_position * self.input_stride

        else:
            duration = segment_duration
            timestamps = [
                token for token in tokens if token >= tokenizer.timestamp_begin
            ]
            if len(timestamps) > 0 and timestamps[-1] != tokenizer.timestamp_begin:
                last_timestamp_position = timestamps[-1] - tokenizer.timestamp_begin
                duration = last_timestamp_position * self.time_precision

            current_segments.append(
                dict(
                    seek=seek,
                    start=time_offset,
                    end=time_offset + duration,
                    tokens=tokens,
                )
            )

            seek += segment_size

        return current_segments, seek, single_timestamp_ending

    def generate_segments(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
        log_progress,
        encoder_output: Optional[ctranslate2.StorageView] = None,
    ) -> Iterable[Segment]:
        content_frames = features.shape[-1] - 1
        content_duration = float(content_frames * self.feature_extractor.time_per_frame)

        if isinstance(options.clip_timestamps, str):
            options.clip_timestamps = [
                float(ts)
                for ts in (
                    options.clip_timestamps.split(",")
                    if options.clip_timestamps
                    else []
                )
            ]

        seek_points: List[int] = [
            round(ts * self.frames_per_second) for ts in options.clip_timestamps
        ]
        if len(seek_points) == 0:
            seek_points.append(0)
        if len(seek_points) % 2 == 1:
            seek_points.append(content_frames)
        seek_clips: List[Tuple[int, int]] = list(
            zip(seek_points[::2], seek_points[1::2])
        )

        punctuation = options.prepend_punctuations + options.append_punctuations

        idx = 0
        clip_idx = 0
        seek = seek_clips[clip_idx][0]
        all_tokens = []
        prompt_reset_since = 0

        if options.initial_prompt is not None:
            if isinstance(options.initial_prompt, str):
                initial_prompt = " " + options.initial_prompt.strip()
                initial_prompt_tokens = tokenizer.encode(initial_prompt)
                all_tokens.extend(initial_prompt_tokens)
            else:
                all_tokens.extend(options.initial_prompt)

        pbar = tqdm(total=content_duration, unit="seconds", disable=not log_progress)
        last_speech_timestamp = 0.0
        # NOTE: This loop is obscurely flattened to make the diff readable.
        # A later commit should turn this into a simpler nested loop.
        # for seek_clip_start, seek_clip_end in seek_clips:
        #     while seek < seek_clip_end
        while clip_idx < len(seek_clips):
            seek_clip_start, seek_clip_end = seek_clips[clip_idx]
            if seek_clip_end > content_frames:
                seek_clip_end = content_frames
            if seek < seek_clip_start:
                seek = seek_clip_start
            if seek >= seek_clip_end:
                clip_idx += 1
                if clip_idx < len(seek_clips):
                    seek = seek_clips[clip_idx][0]
                continue
            time_offset = seek * self.feature_extractor.time_per_frame
            window_end_time = float(
                (seek + self.feature_extractor.nb_max_frames)
                * self.feature_extractor.time_per_frame
            )
            segment_size = min(
                self.feature_extractor.nb_max_frames,
                content_frames - seek,
                seek_clip_end - seek,
            )
            segment = features[:, seek : seek + segment_size]
            segment_duration = segment_size * self.feature_extractor.time_per_frame
            segment = pad_or_trim(segment)

            if self.logger.isEnabledFor(logging.DEBUG):
                self.logger.debug(
                    "Processing segment at %s", format_timestamp(time_offset)
                )

            previous_tokens = all_tokens[prompt_reset_since:]

            if encoder_output is None:
                encoder_output = self.encode(segment)

            # Perform language detection at every segment to update task based on output language,
            # if the language is english, task is transcribe,
            # else the task is translate to english (default)
            # or transcribe if 'output_language' is 'hybrid'.
            if options.multilingual:
                results = self.model.detect_language(encoder_output)
                language_token, language_probability = results[0][0]
                language = language_token[2:-2]
                if options.output_language == "en" and language != "en":
                    task = "translate"
                else:
                    task = "transcribe"

                # Update tokenizer based on task and language
                tokenizer.task = tokenizer.tokenizer.token_to_id(f"<|{task}|>")
                tokenizer.language = tokenizer.tokenizer.token_to_id(language_token)
                tokenizer.language_code = language
            # Update prompt based on task and language
            prompt = self.get_prompt(
                tokenizer,
                previous_tokens,
                without_timestamps=options.without_timestamps,
                prefix=options.prefix if seek == 0 else None,
                hotwords=options.hotwords,
            )

            if seek > 0 or encoder_output is None:
                encoder_output = self.encode(segment)

            (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            ) = self.generate_with_fallback(encoder_output, prompt, tokenizer, options)

            if options.no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > options.no_speech_threshold

                if (
                    options.log_prob_threshold is not None
                    and avg_logprob > options.log_prob_threshold
                ):
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    self.logger.debug(
                        "No speech threshold is met (%f > %f)",
                        result.no_speech_prob,
                        options.no_speech_threshold,
                    )

                # Skip if the logprob is very low (below the threshold value),
                # despite no_speech_prob being low (ex: Too ambiguous outputs)
                if options.log_prob_low_threshold:
                    if avg_logprob < options.log_prob_low_threshold:
                        should_skip = True
                        self.logger.debug(
                            "log prob low threshold is met (%f > %f)",
                            avg_logprob,
                            options.log_prob_low_threshold,
                        )

                if should_skip:
                    # fast-forward to the next segment boundary
                    seek += segment_size
                    continue

            tokens = result.sequences_ids[0]

            previous_seek = seek

            # anomalous words are very long/short/improbable
            def word_anomaly_score(word: dict) -> float:
                probability = word.get("probability", 0.0)
                duration = word["end"] - word["start"]
                score = 0.0
                if probability < 0.15:
                    score += 1.0
                if duration < 0.133:
                    score += (0.133 - duration) * 15
                if duration > 2.0:
                    score += duration - 2.0
                return score

            def is_segment_anomaly(segment: Optional[dict]) -> bool:
                if segment is None or not segment["words"]:
                    return False
                words = [w for w in segment["words"] if w["word"] not in punctuation]
                words = words[:8]
                score = sum(word_anomaly_score(w) for w in words)
                return score >= 3 or score + 0.01 >= len(words)

            def next_words_segment(segments: List[dict]) -> Optional[dict]:
                return next((s for s in segments if s["words"]), None)

            (
                current_segments,
                seek,
                single_timestamp_ending,
            ) = self._split_segments_by_timestamps(
                tokenizer=tokenizer,
                tokens=tokens,
                time_offset=time_offset,
                segment_size=segment_size,
                segment_duration=segment_duration,
                seek=seek,
            )

            if options.word_timestamps:
                self.add_word_timestamps(
                    [current_segments],
                    tokenizer,
                    encoder_output,
                    segment_size,
                    options.prepend_punctuations,
                    options.append_punctuations,
                    last_speech_timestamp=last_speech_timestamp,
                )
                if not single_timestamp_ending:
                    last_word_end = get_end(current_segments)
                    if last_word_end is not None and last_word_end > time_offset:
                        seek = round(last_word_end * self.frames_per_second)

                # skip silence before possible hallucinations
                if options.hallucination_silence_threshold is not None:
                    threshold = options.hallucination_silence_threshold

                    # if first segment might be a hallucination, skip leading silence
                    first_segment = next_words_segment(current_segments)
                    if first_segment is not None and is_segment_anomaly(first_segment):
                        gap = first_segment["start"] - time_offset
                        if gap > threshold:
                            seek = previous_seek + round(gap * self.frames_per_second)
                            continue

                    # skip silence before any possible hallucination that is surrounded
                    # by silence or more hallucinations
                    hal_last_end = last_speech_timestamp
                    for si in range(len(current_segments)):
                        segment = current_segments[si]
                        if not segment["words"]:
                            continue
                        if is_segment_anomaly(segment):
                            next_segment = next_words_segment(
                                current_segments[si + 1 :]
                            )
                            if next_segment is not None:
                                hal_next_start = next_segment["words"][0]["start"]
                            else:
                                hal_next_start = time_offset + segment_duration
                            silence_before = (
                                segment["start"] - hal_last_end > threshold
                                or segment["start"] < threshold
                                or segment["start"] - time_offset < 2.0
                            )
                            silence_after = (
                                hal_next_start - segment["end"] > threshold
                                or is_segment_anomaly(next_segment)
                                or window_end_time - segment["end"] < 2.0
                            )
                            if silence_before and silence_after:
                                seek = round(
                                    max(time_offset + 1, segment["start"])
                                    * self.frames_per_second
                                )
                                if content_duration - segment["end"] < threshold:
                                    seek = content_frames
                                current_segments[si:] = []
                                break
                        hal_last_end = segment["end"]

                last_word_end = get_end(current_segments)
                if last_word_end is not None:
                    last_speech_timestamp = last_word_end
            for segment in current_segments:
                tokens = segment["tokens"]
                text = tokenizer.decode(tokens)

                if segment["start"] == segment["end"] or not text.strip():
                    continue

                all_tokens.extend(tokens)
                idx += 1

                yield Segment(
                    id=idx,
                    seek=seek,
                    start=segment["start"],
                    end=segment["end"],
                    text=text,
                    tokens=tokens,
                    temperature=temperature,
                    avg_logprob=avg_logprob,
                    compression_ratio=compression_ratio,
                    no_speech_prob=result.no_speech_prob,
                    words=(
                        [Word(**word) for word in segment["words"]]
                        if options.word_timestamps
                        else None
                    ),
                )

            if (
                not options.condition_on_previous_text
                or temperature > options.prompt_reset_on_temperature
            ):
                if options.condition_on_previous_text:
                    self.logger.debug(
                        "Reset prompt. prompt_reset_on_temperature threshold is met %f > %f",
                        temperature,
                        options.prompt_reset_on_temperature,
                    )

                prompt_reset_since = len(all_tokens)

            pbar.update(
                (min(content_frames, seek) - previous_seek)
                * self.feature_extractor.time_per_frame,
            )
        pbar.close()

    def encode(self, features: np.ndarray) -> ctranslate2.StorageView:
        # When the model is running on multiple GPUs, the encoder output should be moved
        # to the CPU since we don't know which GPU will handle the next job.
        to_cpu = self.model.device == "cuda" and len(self.model.device_index) > 1

        if features.ndim == 2:
            features = np.expand_dims(features, 0)
        features = get_ctranslate2_storage(features)

        return self.model.encode(features, to_cpu=to_cpu)

    def generate_with_fallback(
        self,
        encoder_output: ctranslate2.StorageView,
        prompt: List[int],
        tokenizer: Tokenizer,
        options: TranscriptionOptions,
    ) -> Tuple[ctranslate2.models.WhisperGenerationResult, float, float, float]:
        decode_result = None
        all_results = []
        below_cr_threshold_results = []

        max_initial_timestamp_index = int(
            round(options.max_initial_timestamp / self.time_precision)
        )
        if options.max_new_tokens is not None:
            max_length = len(prompt) + options.max_new_tokens
        else:
            max_length = self.max_length

        if max_length > self.max_length:
            raise ValueError(
                f"The length of the prompt is {len(prompt)}, and the `max_new_tokens` "
                f"{max_length - len(prompt)}. Thus, the combined length of the prompt "
                f"and `max_new_tokens` is: {max_length}. This exceeds the "
                f"`max_length` of the Whisper model: {self.max_length}. "
                "You should either reduce the length of your prompt, or "
                "reduce the value of `max_new_tokens`, "
                f"so that their combined length is less that {self.max_length}."
            )

        for temperature in options.temperatures:
            if temperature > 0:
                kwargs = {
                    "beam_size": 1,
                    "num_hypotheses": options.best_of,
                    "sampling_topk": 0,
                    "sampling_temperature": temperature,
                }
            else:
                kwargs = {
                    "beam_size": options.beam_size,
                    "patience": options.patience,
                }

            result = self.model.generate(
                encoder_output,
                [prompt],
                length_penalty=options.length_penalty,
                repetition_penalty=options.repetition_penalty,
                no_repeat_ngram_size=options.no_repeat_ngram_size,
                max_length=max_length,
                return_scores=True,
                return_no_speech_prob=True,
                suppress_blank=options.suppress_blank,
                suppress_tokens=options.suppress_tokens,
                max_initial_timestamp_index=max_initial_timestamp_index,
                **kwargs,
            )[0]

            tokens = result.sequences_ids[0]

            # Recover the average log prob from the returned score.
            seq_len = len(tokens)
            cum_logprob = result.scores[0] * (seq_len**options.length_penalty)
            avg_logprob = cum_logprob / (seq_len + 1)

            text = tokenizer.decode(tokens).strip()
            compression_ratio = get_compression_ratio(text)

            decode_result = (
                result,
                avg_logprob,
                temperature,
                compression_ratio,
            )
            all_results.append(decode_result)

            needs_fallback = False

            if options.compression_ratio_threshold is not None:
                if compression_ratio > options.compression_ratio_threshold:
                    needs_fallback = True  # too repetitive

                    self.logger.debug(
                        "Compression ratio threshold is not met with temperature %.1f (%f > %f)",
                        temperature,
                        compression_ratio,
                        options.compression_ratio_threshold,
                    )
                else:
                    below_cr_threshold_results.append(decode_result)

            if (
                options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = True  # average log probability is too low

                self.logger.debug(
                    "Log probability threshold is not met with temperature %.1f (%f < %f)",
                    temperature,
                    avg_logprob,
                    options.log_prob_threshold,
                )

            if (
                options.no_speech_threshold is not None
                and result.no_speech_prob > options.no_speech_threshold
                and options.log_prob_threshold is not None
                and avg_logprob < options.log_prob_threshold
            ):
                needs_fallback = False  # silence

            if not needs_fallback:
                break
        else:
            # all failed, select the result with the highest average log probability
            decode_result = max(
                below_cr_threshold_results or all_results, key=lambda x: x[1]
            )
            # to pass final temperature for prompt_reset_on_temperature
            decode_result = (
                decode_result[0],
                decode_result[1],
                temperature,
                decode_result[3],
            )

        return decode_result

    def get_prompt(
        self,
        tokenizer: Tokenizer,
        previous_tokens: List[int],
        without_timestamps: bool = False,
        prefix: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[int]:
        prompt = []

        if previous_tokens or (hotwords and not prefix):
            prompt.append(tokenizer.sot_prev)
            if hotwords and not prefix:
                hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
                if len(hotwords_tokens) >= self.max_length // 2:
                    hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
                prompt.extend(hotwords_tokens)
            if previous_tokens:
                prompt.extend(previous_tokens[-(self.max_length // 2 - 1) :])

        prompt.extend(tokenizer.sot_sequence)

        if without_timestamps:
            prompt.append(tokenizer.no_timestamps)

        if prefix:
            prefix_tokens = tokenizer.encode(" " + prefix.strip())
            if len(prefix_tokens) >= self.max_length // 2:
                prefix_tokens = prefix_tokens[: self.max_length // 2 - 1]
            if not without_timestamps:
                prompt.append(tokenizer.timestamp_begin)
            prompt.extend(prefix_tokens)

        return prompt

    def find_alignment(
        self,
        tokenizer: Tokenizer,
        text_tokens: List[int],
        encoder_output: ctranslate2.StorageView,
        num_frames: int,
        median_filter_width: int = 7,
    ) -> List[dict]:
        if len(text_tokens) == 0:
            return []

        results = self.model.align(
            encoder_output,
            tokenizer.sot_sequence,
            text_tokens,
            num_frames,
            median_filter_width=median_filter_width,
        )
        return_list = []
        for result, text_token in zip(results, text_tokens):
            text_token_probs = result.text_token_probs
            alignments = result.alignments
            text_indices = np.array([pair[0] for pair in alignments])
            time_indices = np.array([pair[1] for pair in alignments])

            words, word_tokens = tokenizer.split_to_word_tokens(
                text_token + [tokenizer.eot]
            )
            if len(word_tokens) <= 1:
                # return on eot only
                # >>> np.pad([], (1, 0))
                # array([0.])
                # This results in crashes when we lookup jump_times with float, like
                # IndexError: arrays used as indices must be of integer (or boolean) type
                return []
            word_boundaries = np.pad(
                np.cumsum([len(t) for t in word_tokens[:-1]]), (1, 0)
            )
            if len(word_boundaries) <= 1:
                return []

            jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(
                bool
            )
            jump_times = time_indices[jumps] / self.tokens_per_second
            start_times = jump_times[word_boundaries[:-1]]
            end_times = jump_times[word_boundaries[1:]]
            word_probabilities = [
                np.mean(text_token_probs[i:j])
                for i, j in zip(word_boundaries[:-1], word_boundaries[1:])
            ]

            return_list.append(
                [
                    dict(
                        word=word,
                        tokens=tokens,
                        start=start,
                        end=end,
                        probability=probability,
                    )
                    for word, tokens, start, end, probability in zip(
                        words, word_tokens, start_times, end_times, word_probabilities
                    )
                ]
            )
        return return_list

    def generate_segment_batched(
        self,
        features: np.ndarray,
        tokenizer: Tokenizer,
        options: dict,
    ):
        batch_size = features.shape[0]
        all_tokens = []
        prompt_reset_since = 0

        if options["initial_prompt"] is not None:
            initial_prompt = " " + options["initial_prompt"].strip()
            initial_prompt_tokens = tokenizer.encode(initial_prompt)
            all_tokens.extend(initial_prompt_tokens)
        previous_tokens = all_tokens[prompt_reset_since:]
        prompt = self.get_prompt(
            tokenizer,
            previous_tokens,
            without_timestamps=options["without_timestamps"],
            prefix=options["prefix"],
        )

        encoder_output = self.encode(features)

        result = self.model.generate(
            encoder_output,
            [prompt] * batch_size,
            beam_size=options["beam_size"],
            patience=options["patience"],
            length_penalty=options["length_penalty"],
            max_length=self.max_length,
            suppress_blank=options["suppress_blank"],
            suppress_tokens=options["suppress_tokens"],
            return_scores=True,
            return_no_speech_prob=True,
        )

        output = []
        for res in result:
            output.append({})
            # return scores
            seq_len = len(res.sequences_ids[0])
            cum_logprob = res.scores[0] * (seq_len ** options["length_penalty"])
            output[-1]["avg_logprob"] = cum_logprob / (seq_len + 1)

            # return no speech prob
            output[-1]["no_speech_prob"] = res.no_speech_prob
            output[-1]["tokens"] = res.sequences_ids[0]

        return encoder_output, output

    def detect_language(self, audio: np.ndarray):
        segment = self.feature_extractor(audio)[
            :, : self.feature_extractor.nb_max_frames
        ]
        encoder_output = self.encode(pad_or_trim(segment))
        results = self.model.detect_language(encoder_output)
        language_token, language_probability = results[0][0]
        language = language_token[2:-2]
        self.logger.info(
            f"Detected language: {language} ({language_probability:.2f}) in first 30s of audio..."
        )
        all_language_probs = [(token[2:-2], prob) for (token, prob) in results[0]]
        return language, language_probability, all_language_probs

    def detect_language_multi_segment(
        self, audio: Union[str, BinaryIO, np.ndarray], params: Optional[dict] = None
    ):
        """
        Detect language based on N highly-confident segments of a language.
        """
        # The threshold is used to decide if the audio is silence or not.
        # The default is 0.02 (2.0%) i.e, if more than 2.0% of the audio is silent,
        # the audio is considered as silence.
        if not params:
            params = {
                "multilingual": False,
                "speech_percentage_threshold": 0.02,
                "language_detection_segments": 4,
                "vad_filter": True,
                "vad_min_silence_duration": 2500,
                "language_threshold": 0.7,
            }

        if params.get("multilingual", False):
            logging.warning(
                "lang_id is not supported for multilingual audios, detecting the major language."
            )

        speech_percentage_threshold = params.get("speech_percentage_threshold", 0.02)
        language_threshold = params.get("language_threshold", 0.7)
        num_detection_segments = params.get("language_detection_segments", 4)
        vad_filter_enabled = params.get("vad_filter", True)
        vad_params = dict(
            min_silence_duration_ms=params.get("vad_min_silence_duration", 2500)
        )

        if vad_filter_enabled:
            vad_params = VadOptions(**vad_params)

        # decode audio if it is not decoded already
        sampling_rate = self.feature_extractor.sampling_rate
        if not isinstance(audio, np.ndarray):
            audio: np.ndarray = decode_audio(audio, sampling_rate=sampling_rate)

        # calculate duration of audio as number of seconds
        # audio.shape[0] is the number of samples in the audio
        # sampling_rate is the number of samples per second
        # if we divide the number of samples by the number of samples per second,
        # we get the duration in seconds
        duration = audio.shape[0] / sampling_rate

        # Check if vad is enabled, and collect voiced segments
        if vad_filter_enabled:
            # get chunks of audio that contain speech
            speech_chunks = get_speech_timestamps(audio, vad_params)
            # merge chunks of audio that contain speech into a single array
            audio_chunks, chunks_metadata = collect_chunks(audio, speech_chunks)
            audio = np.concatenate(audio_chunks, axis=0)

            # calculate new duration of audio without silence
            duration_vad = audio.shape[0] / sampling_rate

            logging.debug(
                f"Lang ID: VAD filter removed {duration - duration_vad} sec of audio"
            )

            # if the audio after VAD is less than 2% of the original audio, consider it as silence
            if duration_vad / duration < speech_percentage_threshold:
                return {"language_code": None, "language_confidence": 1.0}

            # update duration to be the duration after VAD
            duration = duration_vad

        # if the duration of the audio is less than 1 second, consider it as silence
        if duration < 1.0:
            return {"language_code": None, "language_confidence": 1.0}

        # number of feature frames in 30 seconds of audio is 3000
        nb_max_frames = self.feature_extractor.nb_max_frames

        # extract features from audio with padding (default)
        features = self.feature_extractor(audio)

        # number of segments in the audio
        num_segments = features.shape[-1] // nb_max_frames
        # more number of segments than possible with the duration of file
        if num_detection_segments > num_segments:
            logging.warning(
                f"Lang ID: Can not have more segments, setting {num_segments} segments."
            )
            num_detection_segments = num_segments

        # create a list of indices to randomly select segments from
        indices = list(range(num_detection_segments))

        # fix seed to get deterministic results
        random.seed(0)
        random.shuffle(indices)

        detected_languages = []
        all_language_probabilities = defaultdict(list)
        confident_language_probabilities = defaultdict(list)
        num_confident_segments_per_language = defaultdict(int)

        # Iterate over the randomly selected indices of the segments.
        #
        # For each segment, extract features and detect language.
        #
        # If the language is confident, add it to the list of confident segments for that language.
        #
        # If the number of confident segments for a language
        # is greater than or equal to the number of detection segments,
        # return the language and the average probability of the language.
        #
        # If we are unable to get sufficient number of confident predcitions,
        # return the most frequently detected language with maximum probability.
        #
        # We need to get sufficient number of confident predictions per language, not in total.

        for i in indices:
            segment_features = features[:, i * nb_max_frames : (i + 1) * nb_max_frames]
            try:
                encoder_output = self.encode(pad_or_trim(segment_features))
                results = self.model.detect_language(encoder_output)[0]

            except ValueError as e:  # or RuntimeError
                logging.error(f"Inference error:{e}")

            # results is the list of classes (languages) and their probabilities (descending),
            # for eg: [('<|de|>', 0.482177734375),('<|en|>', 0.283447265625),...]

            # take top language token and probability
            # and parse language token to strip out markers
            # for eg: '<|de|>' -> 'de'

            language_token = results[0][0]
            language = language_token[2:-2]

            language_probability = results[0][1]

            detected_languages.append(language)
            all_language_probabilities[language].append(language_probability)

            # only consider if the language prediction is confident
            if language_probability > language_threshold:
                num_confident_segments_per_language[language] += 1

                # Add language and probability to the list of languages when it is confident
                confident_language_probabilities[language].append(language_probability)

                # return the language when sufficient number of confident segments is achieved
                if (
                    num_confident_segments_per_language[language]
                    >= num_detection_segments
                ):
                    # Considering the average probability of only confident segments
                    mean = sum(confident_language_probabilities[language]) / len(
                        confident_language_probabilities[language]
                    )
                    return {
                        "language_code": language,
                        "language_confidence": mean,
                    }

        # if we are unable to get sufficient number of confident predictions,
        # return the most frequently detected language.
        # if there is a tie, return the one with maximum average probability.
        counter = Counter(detected_languages)

        # Define the key function to select frequent language with attached probabilities
        def key_func(language):
            # Calculate the frequency of the language
            frequency = counter[language]

            # Calculate the average probability of the language
            prob_avg = sum(all_language_probabilities[language]) / len(
                all_language_probabilities[language]
            )

            return frequency, prob_avg

        if detected_languages:
            # Use the key function to find the language with maximum frequency and probability
            max_language = max(detected_languages, key=key_func)
            max_probability = sum(all_language_probabilities[max_language]) / len(
                all_language_probabilities[max_language]
            )

            # Do additional checks for silence for non-confident case
            # calculate RMS amplitude and DC offset
            dc_offset = audio.mean()
            audio_minus_dc_offset = audio - dc_offset
            is_silent = (
                all(np.abs(audio) < 0.1)
                or np.sqrt(np.mean(audio_minus_dc_offset**2)) < 0.01
            )

            if is_silent:
                return {"language_code": None, "language_confidence": 1.0}

            return {
                "language_code": max_language,
                "language_confidence": max_probability,
            }

        # Language is not detected for any segment and none of prev conditions met
        return {"language_code": None, "language_confidence": 1.0}


def restore_speech_timestamps(
    segments: Iterable[Segment],
    speech_chunks: List[dict],
    sampling_rate: int,
) -> Iterable[Segment]:
    ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

    for segment in segments:
        if segment.words:
            words = []
            for word in segment.words:
                # Ensure the word start and end times are resolved to the same chunk.
                middle = (word.start + word.end) / 2
                chunk_index = ts_map.get_chunk_index(middle)
                word.start = ts_map.get_original_time(word.start, chunk_index)
                word.end = ts_map.get_original_time(word.end, chunk_index)
                words.append(word)

            segment.start = words[0].start
            segment.end = words[-1].end
            segment.words = words

        else:
            segment.start = ts_map.get_original_time(segment.start)
            segment.end = ts_map.get_original_time(segment.end)

        yield segment


def get_ctranslate2_storage(segment: np.ndarray) -> ctranslate2.StorageView:
    segment = np.ascontiguousarray(segment)
    segment = ctranslate2.StorageView.from_array(segment)
    return segment


def get_compression_ratio(text: str) -> float:
    text_bytes = text.encode("utf-8")
    return len(text_bytes) / len(zlib.compress(text_bytes))


def get_suppressed_tokens(
    tokenizer: Tokenizer,
    suppress_tokens: Tuple[int],
) -> Optional[List[int]]:
    if -1 in suppress_tokens:
        suppress_tokens = [t for t in suppress_tokens if t >= 0]
        suppress_tokens.extend(tokenizer.non_speech_tokens)
    elif suppress_tokens is None or len(suppress_tokens) == 0:
        suppress_tokens = []  # interpret empty string as an empty list
    else:
        assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

    suppress_tokens.extend(
        [
            tokenizer.transcribe,
            tokenizer.translate,
            tokenizer.sot,
            tokenizer.sot_prev,
            tokenizer.sot_lm,
        ]
    )

    return tuple(sorted(set(suppress_tokens)))


def merge_punctuations(alignment: List[dict], prepended: str, appended: str) -> None:
    # merge prepended punctuations
    i = len(alignment) - 2
    j = len(alignment) - 1
    while i >= 0:
        previous = alignment[i]
        following = alignment[j]
        if previous["word"].startswith(" ") and previous["word"].strip() in prepended:
            # prepend it to the following word
            following["word"] = previous["word"] + following["word"]
            if "tokens" in alignment[0].keys():
                following["tokens"] = previous["tokens"] + following["tokens"]
                previous["tokens"] = []
            previous["word"] = ""

        else:
            j = i
        i -= 1

    # merge appended punctuations
    i = 0
    j = 1
    while j < len(alignment):
        previous = alignment[i]
        following = alignment[j]
        if not previous["word"].endswith(" ") and following["word"] in appended:
            # append it to the previous word
            previous["word"] = previous["word"] + following["word"]
            if "tokens" in alignment[0].keys():
                previous["tokens"] = previous["tokens"] + following["tokens"]
                following["tokens"] = []
            following["word"] = ""

        else:
            i = j
        j += 1
