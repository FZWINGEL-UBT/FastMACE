"""The versioned label table (SPEC §4.3, §7.6, §12.5): label → checkpoint, head, model type, functional, licence.

A label names one checkpoint and, for multi-head checkpoints, one head. The
checkpoint record holds the official release URL, the file name, the SHA-256
of the released file and the licence; a download or a local copy is accepted
only when its SHA-256 matches (``fastmace.model.loading``).

The labels are the research engine's (``common.MODELS`` at ResearchModule tag
``research-engine-baseline-20261017``): the ten benchmark models
(``BENCHMARK_MODELS``), POLAR-1 S/M/L and OMOL-0 XL from the Langevin model
screen, and the five MACE-OFF checkpoints from the organic screen. Together they
are the 19 labels of the benchmark suite (§12.5).

Hashes: SHA-256 of the files in the research engine's model store, which were
downloaded from the URLs below (checked 17 October 2026). Licences: the ACEsuit
``mace-foundations`` model table, the ``mace_polar_1`` release notes ("ASL") and
the ``ACEsuit/mace-off`` repository (ASL), checked 17 October 2026.

This module is data only: it imports nothing heavy and holds no state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType

ASL = "ASL"  # Academic Software License: non-commercial use only
MIT = "MIT"
LICENCE_TEXT = MappingProxyType({
    ASL: "Academic Software License (non-commercial use only), https://github.com/gabor1/ASL",
    MIT: "MIT License",
})

# Model families of SPEC §4.3, as in tests/_support/tiny_models.FAMILIES.
FAMILIES = ("mh1", "omol", "polar", "residual", "density", "off")


@dataclass(frozen=True)
class Checkpoint:
    """One released checkpoint file."""

    name: str  # the research engine's model-store key (model_store.URLS)
    url: str
    sha256: str
    licence: str  # ASL or MIT
    model_type: str  # MACECalculator model_type: "MACE" or "PolarMACE"
    family: str  # one of FAMILIES
    # Heads stored in the checkpoint, in its order, with the training functional of each head
    # (None where the release does not say). Single-head checkpoints have one entry.
    heads: tuple = ()
    head_functionals: MappingProxyType = field(default_factory=lambda: MappingProxyType({}))
    # Joint embedding of per-graph inputs (OMOL-0), as stored in the checkpoint; None otherwise.
    embedding_specs: MappingProxyType | None = None

    @property
    def file_name(self):
        return self.url.rsplit("/", 1)[1]


@dataclass(frozen=True)
class ModelLabel:
    """A label of the table: one checkpoint and one head."""

    label: str
    checkpoint: Checkpoint
    head: str | None  # the head MACECalculator is given; None: the checkpoint's only head
    functional: str | None  # training functional of that head (D3 parameter set, §7.7)

    @property
    def licence(self):
        return self.checkpoint.licence

    @property
    def model_type(self):
        return self.checkpoint.model_type

    @property
    def family(self):
        return self.checkpoint.family


def _frozen(mapping):
    return MappingProxyType({k: MappingProxyType(dict(v)) if isinstance(v, dict) else v
                             for k, v in mapping.items()})


_MP = "https://github.com/ACEsuit/mace-mp/releases/download"
_FOUNDATIONS = "https://github.com/ACEsuit/mace-foundations/releases/download"
_OFF = "https://raw.githubusercontent.com/ACEsuit/mace-off/main"


def _single(name, url, sha256, licence, family, functional, head="default", model_type="MACE", **extra):
    heads = (head,) if head else ()
    return Checkpoint(name=name, url=url, sha256=sha256, licence=licence, model_type=model_type, family=family,
                      heads=heads, head_functionals=_frozen({head or "": functional}), **extra)


CHECKPOINTS = MappingProxyType({c.name: c for c in (
    _single("medium", f"{_MP}/mace_mp_0/2023-12-03-mace-128-L1_epoch-199.model",
            "01bfe22100139f424713cf921144e5509cbe353d67aa9fa1be9c6e1e0ed35845", MIT, "residual", "pbe", head=None),
    _single("large", f"{_MP}/mace_mp_0/MACE_MPtrj_2022.9.model",
            "f80e992b65ab8f88fdf26964511357c022e92704e4d9bcd086652635a8495b32", MIT, "residual", "pbe", head=None),
    _single("medium-0b3", f"{_MP}/mace_mp_0b3/mace-mp-0b3-medium.model",
            "2f2be696351ac9e94fbe01cdfb6f017679acdbd2db7645209ef55fec9826b012", MIT, "density", "pbe"),
    _single("medium-mpa-0", f"{_MP}/mace_mpa_0/mace-mpa-0-medium.model",
            "75428afe3a1d7d8062e19bcaabd5c433623cabf308242ec9fb493e38604fb638", MIT, "density", "pbe"),
    _single("medium-omat-0", f"{_MP}/mace_omat_0/mace-omat-0-medium.model",
            "d4b14be9afa294eebdbe31a0280b26a0fa29715771e978cbfd3ca24e0d90307a", ASL, "density", "pbe"),
    _single("mace-matpes-pbe-0", f"{_FOUNDATIONS}/mace_matpes_0/MACE-matpes-pbe-omat-ft.model",
            "e618ad582b84239905b9c3b77ce6e9ce111b0ecd1533223a1a6aac7a696b8aa0", ASL, "density", "pbe"),
    _single("mace-matpes-r2scan-0", f"{_FOUNDATIONS}/mace_matpes_0/MACE-matpes-r2scan-omat-ft.model",
            "8f147ecffa1d06a696e5648b13b81abda9c96fda9d0fb3faba3cd3fb27d9bba9", ASL, "density", "r2scan"),
    Checkpoint(
        name="mh-1", url=f"{_FOUNDATIONS}/mace_mh_1/mace-mh-1.model",
        sha256="a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47", licence=ASL,
        model_type="MACE", family="mh1",
        heads=("matpes_r2scan", "mp_pbe_refit_add", "spice_wB97M", "oc20_usemppbe", "omol", "omat_pbe"),
        # PBE/r2SCAN/wB97M per the mace-foundations table (§4.3); SPICE is wB97M-D3(BJ), OMol25 wB97M-V.
        head_functionals=_frozen({"matpes_r2scan": "r2scan", "mp_pbe_refit_add": "pbe",
                                  "spice_wB97M": "wb97m-d3bj", "oc20_usemppbe": "pbe", "omol": "wb97m-v",
                                  "omat_pbe": "pbe"})),
    _single("polar-1-s", f"{_FOUNDATIONS}/mace_polar_1/MACE-POLAR-1-S.model",
            "e4495612037b3b3312633182882a38a694ecac9ea0be2b9889ac0b2a84a99510", ASL, "polar", None,
            head="Default", model_type="PolarMACE"),
    _single("polar-1-m", f"{_FOUNDATIONS}/mace_polar_1/MACE-POLAR-1-M.model",
            "fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a", ASL, "polar", None,
            head="Default", model_type="PolarMACE"),
    _single("polar-1-l", f"{_FOUNDATIONS}/mace_polar_1/MACE-POLAR-1-L.model",
            "9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b", ASL, "polar", None,
            head="Default", model_type="PolarMACE"),
    _single("omol-0-xl", f"{_FOUNDATIONS}/mace_omol_0/MACE-omol-0-extra-large-1024.model",
            "9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469", ASL, "omol", "wb97m-v",
            head="omol",
            # Read from the released checkpoint (leto, 17 October 2026): total charge -100..100
            # (offset 100) and total spin 0..100, both one per graph.
            embedding_specs=_frozen({
                "total_spin": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 1024,
                               "num_classes": 101, "offset": 0},
                "total_charge": {"type": "categorical", "per": "graph", "in_dim": 1, "emb_dim": 1024,
                                 "num_classes": 201, "offset": 100},
            })),
    _single("off23-small", f"{_OFF}/mace_off23/MACE-OFF23_small.model",
            "165cce4cfec5a34b9c64d4ebf95de15d71106bb584b7291c8470f0749977c46f", ASL, "off", "wb97m-d3bj", head=None),
    _single("off23-medium", f"{_OFF}/mace_off23/MACE-OFF23_medium.model",
            "4842c52ad210d6e1f84d6cf1ffa70fae25a7e0d755ed55cf223f43913f587db7", ASL, "off", "wb97m-d3bj", head=None),
    _single("off23-large", f"{_OFF}/mace_off23/MACE-OFF23_large.model",
            "a29e397dbf3e7a24ac50a9b0dfc919bd5a62efa346f5895a6237b0950c1d76f4", ASL, "off", "wb97m-d3bj", head=None),
    _single("off23b-medium", f"{_OFF}/mace_off23/MACE-OFF23b_medium.model",
            "871653738a4fbc8124dba1ff5bc595bd0abf7b849a8538c0825dc29ebadf1680", ASL, "off", "wb97m-d3bj", head=None),
    _single("off24-medium", f"{_OFF}/mace_off24/MACE-OFF24_medium.model",
            "e5ccf5837f685899811a68754e7c994393bfd1a81720393b03c643b46c70bc69", ASL, "off", "wb97m-d3bj", head=None),
)})


def _label(label, checkpoint, head=None):
    entry = CHECKPOINTS[checkpoint]
    key = head if head is not None else (entry.heads[0] if entry.heads else "")
    return ModelLabel(label=label, checkpoint=entry, head=head, functional=entry.head_functionals.get(key))


LABELS = MappingProxyType({m.label: m for m in (
    _label("mp0-medium", "medium"),
    _label("mp0-large", "large"),
    _label("mp0b3-medium", "medium-0b3"),
    _label("mpa0-medium", "medium-mpa-0"),
    _label("omat0-medium", "medium-omat-0"),
    _label("matpes-pbe", "mace-matpes-pbe-0"),
    _label("matpes-r2scan", "mace-matpes-r2scan-0"),
    _label("mh1-omat", "mh-1", "omat_pbe"),
    _label("mh1-mp", "mh-1", "mp_pbe_refit_add"),
    _label("mh1-r2scan", "mh-1", "matpes_r2scan"),
    _label("polar1-s", "polar-1-s"),
    _label("polar1-m", "polar-1-m"),
    _label("polar1-l", "polar-1-l"),
    _label("omol0-xl", "omol-0-xl", "omol"),
    _label("off23-small", "off23-small"),
    _label("off23-medium", "off23-medium"),
    _label("off23-large", "off23-large"),
    _label("off23b-medium", "off23b-medium"),
    _label("off24-medium", "off24-medium"),
)})

# The ten models of the research engine's benchmark tables (common.BENCHMARK_MODELS).
BENCHMARK_LABELS = ("mp0-medium", "mp0-large", "mp0b3-medium", "mpa0-medium", "omat0-medium", "matpes-pbe",
                    "matpes-r2scan", "mh1-omat", "mh1-mp", "mh1-r2scan")
MACE_OFF_LABELS = ("off23-small", "off23-medium", "off23-large", "off23b-medium", "off24-medium")
# Short names for a checkpoint whose head is passed separately: fm.load("mh1", head="matpes_r2scan").
ALIASES = MappingProxyType({"mh1": "mh-1", "omol0": "omol-0-xl"})


def find(name, head=None):
    """The ModelLabel for a label, alias or checkpoint name with an optional head; None if it is not one.

    A label fixes its head; a different ``head=`` raises ValueError. An alias or
    checkpoint name takes ``head=``, which must be one of the checkpoint's heads;
    a multi-head checkpoint without ``head=`` raises ValueError listing the heads.
    """
    if name in LABELS:
        entry = LABELS[name]
        if head is not None and head != (entry.head or _only_head(entry.checkpoint)):
            raise ValueError(f"label {name!r} is head {entry.head!r}; drop head={head!r} or use the checkpoint "
                             f"name {entry.checkpoint.name!r}")
        return entry
    checkpoint = CHECKPOINTS.get(ALIASES.get(name, name))
    if checkpoint is None:
        return None
    if head is None:
        if len(checkpoint.heads) > 1:
            raise ValueError(f"{name!r} has several heads; pass head= one of {', '.join(checkpoint.heads)}")
        head_key = _only_head(checkpoint)
    elif checkpoint.heads and head not in checkpoint.heads:
        raise ValueError(f"{name!r} has no head {head!r}; its heads are {', '.join(checkpoint.heads)}")
    else:
        head_key = head
    for entry in LABELS.values():
        if entry.checkpoint is checkpoint and (entry.head or _only_head(checkpoint)) == head_key:
            return entry
    # A head without its own label (MH-1's spice_wB97M, oc20_usemppbe, omol).
    return ModelLabel(label=f"{checkpoint.name}:{head_key}", checkpoint=checkpoint, head=head,
                      functional=checkpoint.head_functionals.get(head_key))


def _only_head(checkpoint):
    return checkpoint.heads[0] if checkpoint.heads else None
