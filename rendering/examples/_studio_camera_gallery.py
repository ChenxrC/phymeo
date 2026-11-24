"""Lossless contact sheets for the fixed and wrist cameras of the manipulation worlds."""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PAPER = '#f4f3ee'
INK = '#192928'
SECONDARY = '#4b605d'
RULE = '#aab9b4'
OBJECTS = {
    'grasp-sphere': 'Red ball',
    'grasp-box': 'Blue box',
    'place-block': 'Green block',
    'place-capsule': 'Yellow capsule',
    'place-compound': 'Orange stepped block',
    'pusht-planar': 'Standard T block',
    'pusht-redundant': 'Wide T block',
    'pusht-cartesian': 'Narrow T block',
}


def _font(size, bold=False):
    faces = (('segoeuib.ttf', 'DejaVuSans-Bold.ttf') if bold
             else ('segoeui.ttf', 'DejaVuSans.ttf'))
    for face in faces:
        try:
            return ImageFont.truetype(face, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _rgb(source):
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            return image.convert('RGB')
    image = source if isinstance(source, Image.Image) else Image.fromarray(source)
    return image if image.mode == 'RGB' else image.convert('RGB')


def _save(canvas, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, format='PNG')
    return path


def build_gallery(images, names, output_path):
    """Show four worlds per group, fixed above wrist. Keys follow the supplied name mapping."""
    available = {k for k, _ in images}
    keys = [k for k in names if k in available]
    if not keys or set(images) != {(k, view) for k in keys for view in ('fixed', 'wrist')}:
        raise ValueError('every displayed world needs a model name and a fixed/wrist image pair')
    first = _rgb(images[keys[0], 'fixed'])
    tw = min(432, max(290, first.width))
    th = max(1, round(tw * first.height / first.width))
    cols = min(4, len(keys))
    groups = (len(keys) + cols - 1) // cols
    margin, gutter, camera_label, header = 30, 16, 66, 86
    group_title, row_gap, group_gap = 65, 12, 26
    group_height = group_title + 2 * th + row_gap
    width = 2 * margin + camera_label + cols * tw + (cols - 1) * gutter
    height = header + groups * group_height + (groups - 1) * group_gap + margin
    canvas = Image.new('RGB', (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    title, item, small, label = _font(32, True), _font(22, True), _font(15), _font(16, True)
    draw.text((margin, 23), 'Manipulation cameras', font=title, fill=INK, anchor='lt')
    draw.text((width-margin, 38), f'{len(keys)} worlds  /  {len(images)} views', font=small,
              fill=SECONDARY, anchor='rm')
    for group in range(groups):
        top = header + group * (group_height + group_gap)
        draw.line((margin, top, width-margin, top), fill=RULE, width=1)
        for row, view in enumerate(('fixed', 'wrist')):
            y = top + group_title + row * (th + row_gap)
            draw.text((margin, y+th//2), view.capitalize(), font=label, fill=SECONDARY, anchor='lm')
        for col, k in enumerate(keys[group*cols:(group+1)*cols]):
            x = margin + camera_label + col*(tw+gutter)
            model = names[k]
            title_text = OBJECTS.get(model, model.replace('-', ' ').capitalize())
            subtitle = f'{model}   /   World {k}' if model in OBJECTS else f'World {k}'
            draw.text((x, top+10), title_text, font=item, fill=INK, anchor='lt')
            draw.text((x, top+38), subtitle, font=small, fill=SECONDARY, anchor='lt')
            for row, view in enumerate(('fixed', 'wrist')):
                image = _rgb(images[k, view])
                scale = min(1., tw / image.width, th / image.height)
                if scale < 1.:
                    size = max(1, round(image.width*scale)), max(1, round(image.height*scale))
                    image = image.resize(size, Image.Resampling.LANCZOS)
                y = top + group_title + row * (th + row_gap)
                canvas.paste(image, (x+(tw-image.width)//2, y+(th-image.height)//2))
    return _save(canvas, output_path)


def build_comparison(before, after, output_path):
    """Place both full-resolution RGB frames side by side, with labels outside the images."""
    left, right = _rgb(before), _rgb(after)
    if left.size != right.size:
        raise ValueError('a same-camera comparison needs equal native image dimensions')
    w, h = left.size
    margin = max(18, min(44, w//28))
    gap = max(12, margin//2)
    width = 2*w + 2*margin + gap
    header, caption = 69, 42
    canvas = Image.new('RGB', (width, header+caption+h+margin), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 19), 'Studio comparison', font=_font(30, True), fill=INK, anchor='lt')
    draw.text((width-margin, 34), f'Same camera  /  {w} × {h}', font=_font(16),
              fill=SECONDARY, anchor='rm')
    draw.line((margin, header, width-margin, header), fill=RULE, width=1)
    label = _font(21, True)
    for col, (text, image) in enumerate((('Before', left), ('After', right))):
        x = margin + col*(w+gap)
        draw.text((x, header+10), text, font=label, fill=INK, anchor='lt')
        canvas.paste(image, (x, header+caption))
    return _save(canvas, output_path)
