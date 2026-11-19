"""Role-resolved object forces from the accepted native MuJoCo-TYX contact prefix."""

from .area_friction_policy import ROLES


class NativeContactPolicyInput:
    fields = (
        "declared",
        "valid_entry",
        "detected_count",
        "active_count",
        "normal_load_n",
        "force_x_n",
        "force_y_n",
        "force_z_n",
    )
    binding = (
        "native-object-contact-v1",
        ROLES,
        fields,
        "accepted-step-input",
        "force-on-object-world",
    )

    def __init__(self, workspace, scenes, device):
        import torch

        models = {world: model for world, model, _, _ in workspace.binding["source"]["worlds"]}
        codes, declarations = [], []
        for world in workspace.binding["world_keys"]:
            geometry = workspace.model_metadata[models[world]]["geom_names"]
            scene = scenes[world]
            by_name = dict.fromkeys(scene.object_geom_names, -1)
            by_name.update({name: index for index, name in enumerate(scene.finger_geom_names)})
            by_name[scene.support_geom_name] = ROLES.index("support")
            row = [by_name.get(name, ROLES.index("other")) for name in geometry]
            codes.append(row)
            declarations.append([index in row for index in range(len(ROLES))])
        width = max(map(len, codes))
        self._codes = torch.tensor(
            [row + [-2] * (width - len(row)) for row in codes], device=device, dtype=torch.long
        )
        self._declared = torch.tensor(declarations, device=device, dtype=torch.bool)

    def encode(self, values):
        import torch

        ids = values["entry_contact_ids"]
        entry = values["entry_meta"][:, :1] > 0
        valid = entry & (ids[:, :, 0] == 1)
        first = self._codes.gather(1, torch.where(valid, ids[:, :, 2], 0).long())
        second = self._codes.gather(1, torch.where(valid, ids[:, :, 3], 0).long())
        is_first = first == -1
        role = torch.where(is_first, second, first)
        object_pair = valid & (is_first ^ (second == -1))
        force = values["entry_contact_wrench_world"][:, :, :3]
        on_object = torch.where(is_first.unsqueeze(-1), -force, force)
        normal = values["entry_contact_data"][:, :, 13]
        rows = []
        for index in range(len(ROLES)):
            selected = object_pair & (role == index)
            active = selected & (ids[:, :, 1] == 1)
            declared = self._declared[:, index]
            counts = torch.stack(
                (declared, declared & entry[:, 0], selected.sum(1), active.sum(1)), dim=1
            )
            load = torch.where(active, normal, 0.0).sum(1, keepdim=True)
            resultant = torch.where(active.unsqueeze(-1), on_object, 0.0).sum(1)
            rows.append(torch.cat((counts.to(force.dtype), load, resultant), dim=1))
        return torch.stack(rows, dim=1)
