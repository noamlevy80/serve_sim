"""Model-weights residency tracking.

The model-weights tracker is the static counterpart of the KV-cache tracker. It
decomposes a model into *weight shards* -- the per-layer attention/Mamba matrices
and, for an FFN, either the dense block or the individual routed experts (plus a
shared-expert and latent-projection shard when present) -- and the global LM head.
For every shard it records the set of memory devices the shard currently
*resides* on.

Unlike KV, weight shards are never "ungenerated": they exist from init (typically
parked on the system NVM) and are moved up to a compute device's first tier when
that device needs them. Residency is tracked by object identity (``id(device)``),
matching the resource arbiter and the KV-cache tracker: two value-equal
:class:`~serve_sim.hardware.MemoryDevice` instances are distinct locations.

This module is pure bookkeeping: it stores no tensors and does no timing. The
event generator and orchestrator consume it to decide which shards must be moved
(and how big the move is) before a layer can compute.
"""

from __future__ import annotations

from dataclasses import dataclass

from .blocks import DenseFFN, GatedDeltaNet, LayeredModel, MambaBlock, MoEFFN
from .hardware import MemoryDevice


@dataclass(frozen=True)
class WeightShard:
    """One atomically-placed piece of a model's weights.

    Attributes:
        index: Position in the owning tracker's shard list (stable identity).
        component: ``"attention"``, ``"mamba"``, ``"ffn"``, ``"expert"``,
            ``"shared_expert"``, ``"latent_proj"`` or ``"lm_head"``.
        bytes: Stored size of the shard.
        layer_index: Layer the shard belongs to (``None`` for global shards such
            as the LM head).
        expert_index: Routed-expert ordinal for ``"expert"`` shards, else ``None``.
        name: Optional human-readable label for logs/reports.
    """

    index: int
    component: str
    bytes: float
    layer_index: int | None = None
    expert_index: int | None = None
    name: str = ""


class ModelWeightsTracker:
    """Residency of one model's weight shards across memory devices."""

    def __init__(self, shards: list[WeightShard]) -> None:
        self._shards: tuple[WeightShard, ...] = tuple(shards)
        # Per shard (by index): id(device) -> device, so value-equal instances
        # stay distinct locations.
        self._devices: list[dict[int, MemoryDevice]] = [
            {} for _ in self._shards
        ]

    # --- structure ----------------------------------------------------------

    def __len__(self) -> int:
        return len(self._shards)

    @property
    def shards(self) -> tuple[WeightShard, ...]:
        """All weight shards in stable index order."""

        return self._shards

    @property
    def total_bytes(self) -> float:
        """Summed stored size of every shard."""

        return sum(shard.bytes for shard in self._shards)

    def shards_for_layer(self, layer_index: int | None) -> list[WeightShard]:
        """Shards belonging to ``layer_index`` (or globals when ``None``)."""

        return [s for s in self._shards if s.layer_index == layer_index]

    def shard_for(
        self,
        component: str,
        layer_index: int | None = None,
        expert_index: int | None = None,
    ) -> WeightShard:
        """The single shard matching a descriptor (raises if not unique)."""

        matches = [
            s
            for s in self._shards
            if s.component == component
            and s.layer_index == layer_index
            and s.expert_index == expert_index
        ]
        if not matches:
            raise KeyError(
                f"no shard for component={component!r} layer={layer_index} "
                f"expert={expert_index}"
            )
        if len(matches) > 1:
            raise KeyError(
                f"shard descriptor is ambiguous: component={component!r} "
                f"layer={layer_index} expert={expert_index}"
            )
        return matches[0]

    # --- residency ----------------------------------------------------------

    def _slot(self, shard: WeightShard) -> dict[int, MemoryDevice]:
        if not (0 <= shard.index < len(self._shards)) or self._shards[shard.index] is not shard:
            raise ValueError("shard does not belong to this tracker")
        return self._devices[shard.index]

    def place(self, shard: WeightShard, device: MemoryDevice) -> None:
        """Record that ``shard`` now resides on ``device``."""

        self._slot(shard)[id(device)] = device

    def place_all(self, device: MemoryDevice) -> None:
        """Place every shard on ``device`` (e.g. the initial NVM load)."""

        for shard in self._shards:
            self._devices[shard.index][id(device)] = device

    def evict(self, shard: WeightShard, device: MemoryDevice) -> None:
        """Drop ``shard``'s copy on ``device`` (no-op if absent)."""

        self._slot(shard).pop(id(device), None)

    def is_resident(self, shard: WeightShard, device: MemoryDevice) -> bool:
        """Whether ``shard`` currently has a copy on ``device``."""

        return id(device) in self._slot(shard)

    def devices_of(self, shard: WeightShard) -> list[MemoryDevice]:
        """The memory devices ``shard`` currently resides on."""

        return list(self._slot(shard).values())

    def resident_shards(self, device: MemoryDevice) -> list[WeightShard]:
        """Every shard with a copy on ``device``."""

        target = id(device)
        return [s for s in self._shards if target in self._devices[s.index]]

    def bytes_on(self, device: MemoryDevice) -> float:
        """Summed size of the shards currently resident on ``device``."""

        target = id(device)
        return sum(
            s.bytes for s in self._shards if target in self._devices[s.index]
        )

    # --- construction -------------------------------------------------------

    @classmethod
    def from_model(cls, model) -> "ModelWeightsTracker":
        """Enumerate the weight shards of ``model`` (flat or layered)."""

        model = LayeredModel.from_model(model)
        pdb = model.param_dtype_bytes
        shards: list[WeightShard] = []

        def add(
            component: str,
            num_bytes: float,
            layer_index: int | None = None,
            expert_index: int | None = None,
            name: str = "",
        ) -> None:
            shards.append(
                WeightShard(
                    index=len(shards),
                    component=component,
                    bytes=float(num_bytes),
                    layer_index=layer_index,
                    expert_index=expert_index,
                    name=name,
                )
            )

        for li, layer in enumerate(model.layers):
            mixer = layer.mixer
            if mixer is not None:
                if isinstance(mixer, MambaBlock):
                    component = "mamba"
                elif isinstance(mixer, GatedDeltaNet):
                    component = "linear_attention"
                else:
                    component = "attention"
                add(component, mixer.weight_params * pdb, layer_index=li,
                    name=f"layer{li}.{component}")
            ffn = layer.ffn
            if isinstance(ffn, MoEFFN):
                for e in range(ffn.num_experts):
                    add("expert", ffn.routed_expert_params * pdb, layer_index=li,
                        expert_index=e, name=f"layer{li}.expert{e}")
                if ffn.shared_expert_params > 0:
                    add("shared_expert", ffn.shared_expert_params * pdb,
                        layer_index=li, name=f"layer{li}.shared_expert")
                if ffn.latent_proj_params > 0:
                    add("latent_proj", ffn.latent_proj_params * pdb,
                        layer_index=li, name=f"layer{li}.latent_proj")
            elif isinstance(ffn, DenseFFN):
                add("ffn", ffn.weight_params * pdb, layer_index=li,
                    name=f"layer{li}.ffn")

        if model.lm_head_params > 0:
            add("lm_head", model.lm_head_bytes, name="lm_head")

        return cls(shards)
