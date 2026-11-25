# Breakfast nook photographic assets

The home showcase includes CC0 photography and surface scans published by Poly Haven. The scene can be assembled offline from the files beside this notice. The ten household models have separate source and material records in the film manifest. [Poly Haven asset license](https://polyhaven.com/license)

## Window, wall print and lighting

Dimitrios Savva photographed **Garden Nook** and Jarod Guest processed it. The window frames a rectilinear view of the lawn, trees and house. The wall print is a monochrome view looking up into the canopy at the same site. The HDR panorama supplies environmental lighting and reflections alongside the room's area lights. [Garden Nook](https://polyhaven.com/a/garden_nook) · [Published download manifest](https://api.polyhaven.com/files/garden_nook)

| Included file | Use | SHA256 |
| --- | --- | --- |
| `garden_nook_1k.hdr` | Original linear environment panorama, 1,836,222 bytes | `f8c7970483be873487fabc0f07891be7e1551220dba911ea9dadd56f8a336a00` |
| `garden_nook_window_1600.jpg` | Perspective window photograph, 1600 × 900 | `ab26d2c8109f4749d5c6cdb7f2e289b501e49a7d5c78dc0e639e3bd07731e2d3` |
| `garden_nook_canopy_print.jpg` | Perspective monochrome photograph, 544 × 688 | `b5efe0a41488534c6df8813c3c1e771b0c513c1f82103efcc63e52f43be0e6b4` |

Both photographs derive from the publisher's tone-mapped panorama, whose MD5 is `e9db80ff4705ea7c03942c3a5bbf3619`. FFmpeg 8.1.2 decoded the JPEG with `-lowres 2` and projected it with the `v360` equirectangular-to-flat Lanczos filter. The window uses yaw −48°, pitch 4°, horizontal/vertical fields of view 82°/52°. The print uses yaw 47°, pitch 32°, fields of view 42°/50°, saturation 0, contrast 0.88 and brightness 0.035. Both JPEGs use quality setting 2. [Source panorama](https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/garden_nook.jpg) · [Source HDR](https://dl.polyhaven.org/file/ph-assets/HDRIs/hdr/1k/garden_nook_1k.hdr)

The 1024 × 512 RGBE light map contains a resolved solar disc. Its strongest original channel is 120,320. The example uses a uniform gain of 0.00065, which places this channel at 78.208 within the renderer's 100-unit limit. The RGBE file retains its original bytes. The scene's area lights and directional lights illuminate the room separately from the display-referred window photograph.

## Interior finishes

| Scan and creator | Included file | SHA256 |
| --- | --- | --- |
| Wood Floor, Dimitrios Savva | `wood_floor_diff_1k.jpg` | `2a1c07687b6dbb214c4b9213739c6d92d425f1f0fc8ab3ac157f105d78240789` |
| Wood Floor, Dimitrios Savva | `wood_floor_rough_1k.jpg` | `061f1e1293251b2d28c76e3fac1ea9452b0e8648bb8f9b3728e09db386166e5c` |
| White Plaster 02, Rob Tuytel | `white_plaster_02_diff_1k.jpg` | `a1ebbe091bd1ae93d2abd5de8d69f9003a8d0ee6532bcf9a87c2492c97051f23` |

The original floor color and roughness maps share a 1.70 m repeat. The plaster scan uses a 1 m repeat. The scene models white limewash by applying the scan's local luminance variation to an off-white pigment. [Wood Floor](https://polyhaven.com/a/wood_floor) · [Original floor color](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_floor/wood_floor_diff_1k.jpg) · [Original floor roughness](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/wood_floor/wood_floor_rough_1k.jpg) · [White Plaster 02](https://polyhaven.com/a/white_plaster_02) · [Original plaster scan](https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/white_plaster_02/white_plaster_02_diff_1k.jpg)

Furniture surfaces share the previously included **Wood Table 001** color and roughness scans. The table displays the color scan. The chair spindles, shelf, bench and cabinet use tinted variations of its measured grain and mapped roughness. [Existing scan credits and checksums](NOTICE.md)
