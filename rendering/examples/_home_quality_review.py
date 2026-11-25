"""Compare matching numbered stills without starting either renderer.

python _home_quality_review.py --old old/ --new new/ --frame 345 --out review/
For motion, supply contiguous lossless exposures and add --count 6. Custom
--roi 'Subject,x,y,w,h' coordinates refer to the 1440 x 810 review canvas.
"""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DISPLAY = (1440, 810)
ORIGINAL = {
    0: '0089114f16b1f90fd97fecdc5df133b484c772ffccf5bd366238967cf99cc926',
    345: 'ac01250d2146489e3691be4522378245a94d05bfb8608e7dc6d3748f5021e749',
}
REGIONS = {
    0: [('Oblique tabletop edge', 160, 482, 150, 88),
        ('Numerals and clock housing', 1093, 471, 134, 87),
        ('Window frame and foliage', 445, 179, 145, 87)],
    345: [('Silhouette of plate', 379, 430, 155, 98),
          ('Food and plate shading', 578, 567, 155, 98),
          ('Chair against the wall', 364, 0, 155, 98)],
}


def _font(size):
    for path in ('C:/Windows/Fonts/segoeui.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def _region(value):
    try:
        title, *numbers = value.rsplit(',', 4)
        x, y, w, h = map(int, numbers)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('use Subject,x,y,w,h') from exc
    if not title or min(x, y) < 0 or min(w, h) <= 0 or x+w > DISPLAY[0] or y+h > DISPLAY[1]:
        raise argparse.ArgumentTypeError('crop must fit the 1440 x 810 review canvas')
    return title, x, y, w, h


def _read(directory, number):
    stem = f'frame-{number:04d}'
    path = next((directory/(stem+suffix) for suffix in ('.png', '.jpg', '.jpeg')
                 if (directory/(stem+suffix)).is_file()), None)
    if path is None:
        raise FileNotFoundError(f'{directory}/{stem}.png or .jpg')
    digest = sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as source:
        if source.width * DISPLAY[1] != source.height * DISPLAY[0]:
            raise ValueError(f'{path}: camera aspect ratio differs from 16:9')
        metadata = dict(file=str(path.resolve()), format=source.format, size=source.size, sha256=digest,
                        original_film_jpeg=digest == ORIGINAL.get(number))
        pixels = source.convert('RGB')
    if pixels.size != DISPLAY:
        pixels = pixels.resize(DISPLAY, Image.Resampling.LANCZOS)
    return pixels, metadata


def _sheet(images, metadata, regions, tick, matched, labels):
    zoom = 3
    cell = max(505, 40 + zoom*max(r[3] for r in regions))
    heights = [52+zoom*r[4] for r in regions]
    thumb = (cell-40, round((cell-40)*DISPLAY[1]/DISPLAY[0]))
    start = 126+thumb[1]
    sheet = Image.new('RGB', (2*cell, start+sum(heights)+14), '#f3f3ef')
    draw = ImageDraw.Draw(sheet)
    heading, text = _font(20), _font(15)
    draw.text((20, 10), f'Home showcase  /  frame {tick:04d}  /  {tick/30:.3f} s', font=heading, fill='#18292b')
    title = ('Same scene and camera, supplied by caller' if matched else
             'Appearance comparison  /  lighting, materials or scene may differ')
    draw.text((20, 39), f'{title}.  Display: 1440 x 810.  Crops: 3x nearest.', font=text, fill='#34464a')
    for side, (image, meta, label) in enumerate(zip(images, metadata, labels, strict=True)):
        left = side*cell+20
        tag = 'archived original JPEG' if meta['original_film_jpeg'] else meta['format']
        draw.text((left, 72), f'{label}   |   {meta["size"][0]} x {meta["size"][1]}   |   {tag}',
                  font=heading, fill='#18292b')
        small = image.resize(thumb, Image.Resampling.LANCZOS)
        marks = ImageDraw.Draw(small)
        for n, (_, x, y, w, h) in enumerate(regions, 1):
            x0, y0 = x*thumb[0]/DISPLAY[0], y*thumb[1]/DISPLAY[1]
            x1, y1 = (x+w)*thumb[0]/DISPLAY[0], (y+h)*thumb[1]/DISPLAY[1]
            marks.rectangle((x0, y0, x1, y1), outline='#ffcf52', width=2)
            marks.text((x0+2,y0+1), str(n), fill='black', stroke_fill='white', stroke_width=1, font=text)
        sheet.paste(small, (left, 106))
        top = start
        for n, (title, x, y, w, h) in enumerate(regions, 1):
            draw.text((left, top), f'{n}   {title}', font=heading, fill='#18292b')
            crop = image.crop((x, y, x+w, y+h)).resize((w*zoom, h*zoom), Image.Resampling.NEAREST)
            sheet.paste(crop, (left, top+29))
            top += heights[n-1]
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('old', 'new', 'out'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--count', type=int, default=1, help='1..16 adjacent 30 Hz exposures')
    parser.add_argument('--roi', action='append', type=_region)
    parser.add_argument('--labels', nargs=2, default=('Previous', 'Updated'))
    parser.add_argument('--matched', action='store_true',
                        help='caller confirms equal geometry, camera, exposure, material, light at every exposure')
    args = parser.parse_args()
    if not 1 <= args.count <= 16 or args.frame < 0:
        parser.error('expected a nonnegative frame and 1..16 consecutive samples')
    regions = args.roi or REGIONS.get(args.frame)
    if not regions:
        parser.error('supply --roi when starting at a frame other than 0 or 345')
    args.out.mkdir(parents=True, exist_ok=True)
    review = dict(first_frame=args.frame, fps=30, crops=regions, common_display=DISPLAY,
                  matched_by_caller=args.matched, labels=args.labels, frames=[])
    frames = []
    for tick in range(args.frame, args.frame+args.count):
        images, metadata = zip(*(_read(folder, tick) for folder in (args.old, args.new)), strict=True)
        if args.matched and any(m['format'] != 'PNG' for m in metadata):
            raise ValueError('use original lossless PNG render outputs for a controlled AA comparison')
        frames.append(_sheet(images, metadata, regions, tick, args.matched, args.labels))
        review['frames'].append(dict(number=tick, previous=metadata[0], updated=metadata[1]))
    prefix = args.out / f'comparison-{args.frame:04d}'
    frames[0].save(prefix.with_suffix('.png'))
    if len(frames) > 1:
        # Lossless RGB animation; slow both sequences equally for inspection.
        frames[0].save(prefix.with_name(prefix.name+'-motion.png'), save_all=True,
                       append_images=frames[1:], duration=200, loop=0, format='PNG')
        review['review_animation_fps'] = 5
    prefix.with_suffix('.json').write_text(json.dumps(review, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
