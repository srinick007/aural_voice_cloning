import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
import torchaudio
from simple_parsing import ArgumentParser, field

from tortoise.api import MODELS_DIR, TextToSpeech
from tortoise.utils.audio import load_audio
from tortoise.utils.diffusion import SAMPLERS
from tortoise.models.vocoder import VocConf

@dataclass
class Tuning:
    """Tuning options (overrides preset settings)"""

    num_autoregressive_samples: Optional[int] = 2
    """Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
    As TorToiSe is a probabilistic model, more samples means a higher probability of creating something "great"."""

    temperature: Optional[float] = None
    """The softmax temperature of the autoregressive model."""

    length_penalty: Optional[float] = None
    """A length penalty applied to the autoregressive decoder. Higher settings causes the model to produce more terse outputs."""

    repetition_penalty: Optional[float] = None
    """A penalty that prevents the autoregressive decoder from repeating itself during decoding.
    Can be used to reduce the incidence of long silences or "uhhhhhhs", etc."""

    top_p: Optional[float] = None
    """P value used in nucleus sampling. 0 to 1. Lower values mean the decoder produces more "likely" (aka boring) outputs."""

    max_mel_tokens: Optional[int] = None
    """Restricts the output length. 1 to 600. Each unit is 1/20 of a second."""

    cvvp_amount: Optional[float] = None
    """How much the CVVP model should influence the output.
    Increasing this can in some cases reduce the likelihood of multiple speakers."""

    diffusion_iterations: Optional[int] = None
    """Number of diffusion steps to perform.  More steps means the network has more chances to iteratively
    refine the output, which should theoretically mean a higher quality output.
    Generally a value above 250 is not noticeably better, however."""

    cond_free: Optional[bool] = None
    """Whether or not to perform conditioning-free diffusion. Conditioning-free diffusion performs two forward passes for
    each diffusion step: one with the outputs of the autoregressive model and one with no conditioning priors. The output
    of the two is blended according to the cond_free_k value below. Conditioning-free diffusion is the real deal, and
    dramatically improves realism."""

    cond_free_k: Optional[float] = None
    """Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf].
    As cond_free_k increases, the output becomes dominated by the conditioning-free signal.
    Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k"""

    diffusion_temperature: Optional[float] = None
    """Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
    are the "mean" prediction of the diffusion network and will sound bland and smeared."""


@dataclass
class Speed:
    """New/speed options"""

    low_vram: bool = False
    """re-enable default offloading behaviour of tortoise"""

    half: bool = False
    """enable autocast to half precision for autoregressive model"""

    no_cache: bool = False
    """disable kv_cache usage. This should really only be used if you are very low on vram."""

    sampler: Optional[str] = field(default="dpm++2m", choices=SAMPLERS)
    """override the sampler used for diffusion (default depends on --preset)"""

    original_tortoise: bool = False
    """ensure results are identical to original tortoise-tts repo"""



parser = ArgumentParser(
    description="TorToiSe is a text-to-speech program that is capable of synthesizing speech "
    "in multiple voices with realistic prosody and intonation."
)


parser.add_arguments(Tuning, "tuning")
parser.add_arguments(Speed, "speed")

# show usage even when Ctrl+C is pressed early
try:
    args = parser.parse_args()
except SystemExit as e:
    if e.code == 0:
        print(usage_examples)
    sys.exit(e.code)

voices_dir = None
candidates = 1
regenerate = None
skip_existing =  False
produce_debug_state = False
seed = 42
models_dir="../models"
output_path ="../results"
voice = "obama"
batch_size = None
device = None
ar_checkpoint = None
clvp_checkpoint = None

text_split = None
verbose = not False
voicefixer = True
list_voices = False
play = False
diff_checkpoint = None
disable_redaction = False

from tortoise.inference import (
    check_pydub,
    get_all_voices,
    get_seed,
    parse_multiarg_text,
    parse_voice_str,
    split_text,
    validate_output_dir,
    voice_loader,
    save_gen_with_voicefix
)

# get voices
all_voices, extra_voice_dirs = get_all_voices(voices_dir)
if list_voices:
    for v in all_voices:
        print(v)
    sys.exit(0)

selected_voices = parse_voice_str(voice, all_voices)
voice_generator = voice_loader(selected_voices, extra_voice_dirs)
print(selected_voices)

text = "Yesterday is history, tomorrow is a mystery, but today is a gift. That is why it is called the present. There are no accidents. One often meets his destiny on the road he takes to avoid it."
texts = split_text(text, text_split)


output_dir = validate_output_dir(
    output_path, selected_voices, candidates
)
print(output_dir)
# error out early if pydub isn't installed
pydub = check_pydub(play)

seed = get_seed(seed)


vocoder = getattr(VocConf, "BigVGAN_Base")
if verbose:
    print("Loading tts...")
tts = TextToSpeech(
    models_dir=models_dir,
    enable_redaction=disable_redaction,
    device=device,
    autoregressive_batch_size=batch_size,
    high_vram=not args.speed.low_vram,
    kv_cache=not args.speed.no_cache,
    ar_checkpoint=ar_checkpoint,
    clvp_checkpoint=clvp_checkpoint,
    diff_checkpoint=diff_checkpoint,
    vocoder=vocoder,
)

gen_settings = {
    "use_deterministic_seed": seed,
    "verbose": verbose,
    "k": candidates,
    "preset": "fast",
}

tuning_options = [
    "num_autoregressive_samples",
    "temperature",
    "length_penalty",
    "repetition_penalty",
    "top_p",
    "max_mel_tokens",
    "cvvp_amount",
    "diffusion_iterations",
    "cond_free",
    "cond_free_k",
    "diffusion_temperature",
]

for option in tuning_options:
    if getattr(args.tuning, option) is not None:
        gen_settings[option] = getattr(args.tuning, option)

speed_options = [
    "sampler",
    "original_tortoise",
    "half",
]
for option in speed_options:
    if getattr(args.speed, option) is not None:
        gen_settings[option] = getattr(args.speed, option)

total_clips = len(texts) * len(selected_voices)
regenerate_clips = (
    [int(x) for x in regenerate.split(",")]
    if regenerate
    else None
)
for voice_idx, (voice, voice_samples, conditioning_latents) in enumerate(
    voice_generator
):
    audio_parts = []
    for text_idx, text in enumerate(texts):
        clip_name = f'{"-".join(voice)}_{text_idx:02d}'
        if output_dir:
            first_clip = os.path.join(output_path, f"{clip_name}_00.wav")
            if (
                skip_existing
                or (regenerate_clips and text_idx not in regenerate_clips)
            ) and os.path.exists(first_clip):
                audio_parts.append(load_audio(first_clip, 24000))
                if verbose:
                    print(f"Skipping {clip_name}")
                continue
        if verbose:
            print(
                f"Rendering {clip_name} ({(voice_idx * len(texts) + text_idx + 1)} of {total_clips})..."
            )
            print("  " + text)
        gen = tts.tts_with_preset(
            text,
            voice_samples=voice_samples,
            conditioning_latents=conditioning_latents,
            **gen_settings,
        )
        gen = gen if candidates > 1 else [gen]
        for candidate_idx, audio in enumerate(gen):
            audio = audio.squeeze(0).cpu()
            if candidate_idx == 0:
                audio_parts.append(audio)
            if output_path:
                filename = f"{clip_name}_{candidate_idx:02d}.wav"
                save_gen_with_voicefix(audio, os.path.join(output_path, filename), squeeze=False, voicefixer=voicefixer)

    audio = torch.cat(audio_parts, dim=-1)
    if output_path:
        filename = f'{"-".join(voice)}_combined.wav'
        save_gen_with_voicefix(
            audio,
            os.path.join(output_path, filename),
            squeeze=False,
            voicefixer=voicefixer,
        )
