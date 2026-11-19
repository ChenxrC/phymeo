"""CUDA actor features for the object in a manipulation Scene.

One row per physical role. Fn sums compressive sample loads in N, Ft is the
world-frame tangential resultant on the declared object. Multiple patches of
one role add, even when the object occupies different sides. Entry validity
and geometric presence distinguish reset, separation and unloaded contact.
"""

ROLES = ("finger0", "finger1", "support", "other")
FIELDS = (
    "declared",
    "friction_law",
    "entry_valid",
    "has_samples",
    "normal_load_n",
    "tangent_x_n",
    "tangent_y_n",
    "tangent_z_n",
)


class AreaFrictionPolicyInput:
    """Compile validated per-world role declarations into fixed device lookups."""

    binding = ("area-object-contact-v1", ROLES, FIELDS, "accepted-step-input")

    def __init__(self, binding, roles, device):
        import torch

        models = {world: model for world, model, _, _ in binding["source"]["worlds"]}
        width = max(len(pairs) for pairs in binding["area_pair_names"].values())
        pair_codes, declarations, laws = [], [], []
        for world in binding["world_keys"]:
            model = models[world]
            codes = [0] * width
            for i, role in enumerate(roles.get(model, ())):
                if role.object_side is not None:
                    sign = 1 if role.object_side == "A" else -1
                    codes[i] = sign * (1 + ROLES.index(role.role))
            declared = [any(abs(code) == i for code in codes) for i in range(1, 1 + len(ROLES))]
            pair_codes.append(codes)
            declarations.append(declared)
            laws.append(
                [present and binding["area_laws"][model] is not None for present in declared]
            )
        self._codes = torch.tensor(pair_codes, device=device, dtype=torch.long)
        self._declared = torch.tensor(declarations, device=device, dtype=torch.bool)
        self._law = torch.tensor(laws, device=device, dtype=torch.bool)

    def encode(self, values):
        import torch

        ids, data = values["area_ids"], values["area_data"]
        valid_entry = values["area_meta"][:, 0:1] > 0
        codes = self._codes.gather(1, ids[:, :, 1].long())
        valid_sample = valid_entry & (ids[:, :, 0] == 1)
        tangent = data[:, :, 15:18] * codes.sign().unsqueeze(-1)
        rows = []
        for i in range(len(ROLES)):
            selected = valid_sample & (codes.abs() == i + 1)
            declared = self._declared[:, i]
            masks = torch.stack(
                (declared, self._law[:, i], declared & valid_entry[:, 0], selected.any(dim=1)),
                dim=1,
            )
            fn = (data[:, :, 8] * selected).sum(dim=1, keepdim=True)
            ft = (tangent * selected.unsqueeze(-1)).sum(dim=1)
            rows.append(torch.cat((masks.to(data.dtype), fn, ft), dim=1))
        return torch.stack(rows, dim=1)
