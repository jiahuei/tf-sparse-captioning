# -*- coding: utf-8 -*-
"""
Created on 28 Aug 2019 17:15:59

@author: jiahuei
"""
from link_dirs import pjoin
import os
import json
import pickle
import random
import textwrap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fmng
from PIL import Image, ImageEnhance, ImageOps, ImageFont, ImageDraw

# Variables
sort_by_metric = 'CIDEr'
jump_to_idx = 0
model_names = ['sps_80.0', 'sps_97.5']
baseline_name = 'baseline'
output_dir = '/home/jiahuei/Documents/1_TF_files/prune/compiled_mscoco_val'
image_dir = '/master/datasets/mscoco/val2014'
JSON_ROOT = '/home/jiahuei/Documents/1_TF_files/prune'

baseline_json = pjoin(
    JSON_ROOT,
    'mscoco_v2/word_w256_LSTM_r512_cnnFT_SCST_b7C1.0B2.0/run_01___infer_valid_b3_lp0.0___08-27_18-10/captions___113287.json')
model_json = [
    pjoin(JSON_ROOT,
          'mscoco_v3/word_w256_LSTM_r512_h1_ind_xu_REG_1.0e+02_init_5.0_L1_wg_7.5_ann_sps_0.800_dec_prune_cnnFT_SCST_b7C1.0B2.0/run_01___infer_valid_b3_lp0.0___05-23_18-43/captions___113287.json'),
    pjoin(JSON_ROOT,
          'mscoco_v3/word_w256_LSTM_r512_h1_ind_xu_REG_1.0e+02_init_5.0_L1_wg_60.0_ann_sps_0.975_dec_prune_cnnFT_SCST_b7C1.0B2.0/run_01___infer_valid_b3_lp0.0___05-24_07-15/captions___113287.json')]

baseline_scores_json = pjoin(JSON_ROOT,
                             'mscoco_v2/word_w256_LSTM_r512_cnnFT_SCST_b7C1.0B2.0/run_01___infer_valid_b3_lp0.0___08-27_18-10/metric_scores_detailed_113287.json')
model_scores_json = [
    pjoin(JSON_ROOT,
          'mscoco_v3/word_w256_LSTM_r512_h1_ind_xu_REG_1.0e+02_init_5.0_L1_wg_7.5_ann_sps_0.800_dec_prune_cnnFT_SCST_b7C1.0B2.0/run_01___infer_valid_b3_lp0.0___05-23_18-43/metric_scores_detailed_113287.json'),
    pjoin(JSON_ROOT,
          'mscoco_v3/word_w256_LSTM_r512_h1_ind_xu_REG_1.0e+02_init_5.0_L1_wg_60.0_ann_sps_0.975_dec_prune_cnnFT_SCST_b7C1.0B2.0/run_01___infer_valid_b3_lp0.0___05-24_07-15/metric_scores_detailed_113287.json')]

shortlisted_imgs = [
    '0-010_000000431954.jpg',
    '0-015_000000242615.jpg',
    '0-018_000000396608.jpg',
    '0-030_000000105918.jpg',
    '0-073_000000312544.jpg',
    '0-176_000000423935.jpg',
    '0-200_000000538844.jpg',
    '0-267_000000232654.jpg',
    '0-337_000000477750.jpg',
    '0-421_000000531244.jpg',
    '0-558_000000450886.jpg',
    '0-620_000000243896.jpg',
    '0-777_000000075719.jpg',
    '0-805_000000029306.jpg',
    '0-851_000000229599.jpg',
    '0-884_000000025096.jpg',
    '0-893_000000554273.jpg',
    '0-894_000000118839.jpg',
    '0-917_000000261824.jpg',
    '0-954_000000230561.jpg',
    '0-968_000000120872.jpg',
    '1-006_000000169226.jpg',
    '1-050_000000171335.jpg',
    '1-081_000000532620.jpg',
    '1-111_000000209015.jpg',
    '1-160_000000457848.jpg',
    '1-209_000000126634.jpg',
    '1-217_000000387463.jpg',
    '1-234_000000306212.jpg',
    '1-239_000000556101.jpg',
    '1-253_000000023879.jpg',
    '1-254_000000204757.jpg',
    '1-260_000000442688.jpg',
    '1-293_000000125850.jpg',
    '1-354_000000118740.jpg',
    '1-389_000000189446.jpg',
    '1-481_000000335861.jpg',
    '1-492_000000118911.jpg',
    '1-545_000000320857.jpg',
    '1-809_000000332113.jpg',
    '1-830_000000038837.jpg',
    '1-881_000000458401.jpg',
    '1-940_000000104495.jpg',
    '2-166_000000129800.jpg',
    '2-255_000000511674.jpg',
    '2-653_000000397375.jpg',
    '2-710_000000543402.jpg',
    '2-789_000000278977.jpg',
    '2-975_000000542234.jpg',
    '4-218_000000247576.jpg',
]

# Constants
random.seed(3310)
CATEGORIES = dict(
    x='both_wrong',
    y='both_correct',
    b='baseline_correct',
    m='model_correct',
    a='ambiguous',
)
METRICS = ['CIDEr', 'Bleu_4', 'Bleu_3', 'Bleu_2', 'Bleu_1', 'ROUGE_L', 'METEOR']
IMG_RESIZE = 512
IMG_CROP = int(224 / 256 * IMG_RESIZE)
DISPLAY_BG_SIZE = [int(IMG_RESIZE * 4.5), int(IMG_RESIZE * 2.5)]
TEXT_SIZE = int(IMG_RESIZE / 7)
try:
    font = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', TEXT_SIZE)
except OSError:
    FONT_LIST = [f for f in fmng.findSystemFonts(fontpaths=None, fontext='ttf')
                 if 'mono' in os.path.basename(f).lower()]
    font = ImageFont.truetype(FONT_LIST[0], TEXT_SIZE)


def _load_caption_json(res_dict, json_path, name):
    with open(json_path, 'r') as ff:
        captions = json.load(ff)
    for c in captions:
        img_id = c['image_id']
        if type(img_id) == str:
            # Insta-1.1M
            img_name = img_id
        else:
            img_name = 'COCO_val2014_{:012d}.jpg'.format(img_id)
        if img_id not in res_dict:
            res_dict[img_id] = dict(image_id=img_id, image_name=img_name)
        res_dict[img_id][name] = dict(caption=c['caption'])


def _load_score_json(res_dict, json_path, name):
    with open(json_path, 'r') as ff:
        scores = json.load(ff)
    for sc in scores:
        img_id = sc['image_id']
        assert img_id in res_dict
        for m in METRICS:
            res_dict[img_id][name][m] = sc[m]


def _sort_captions(res_dict, sort_metric, sort_model, use_diff=False):
    """
    Return a list of sorted captions.
    :param res_dict: id_to_results
    :param sort_metric: Metric used to sort. If `random`, return list with randomised order.
    :param sort_model: Model result used to sort.
    :param use_diff: If True, use the difference in score between model and baseline to sort.
    :return: A list of sorted captions.
    """
    if isinstance(sort_model, list):
        assert len(sort_model) > 0
        if len(sort_model) > 1:
            pass
        elif len(sort_model) == 1:
            pass
    else:
        sort_model = [sort_model]
    res = list(res_dict.values())
    if sort_metric in METRICS:
        def _get_model_mean(elem):
            sc_m = [elem[m][sort_metric] for m in sort_model]
            return sum(sc_m) / len(sc_m)

        if use_diff:
            def _key_fn(elem):
                sc_m = _get_model_mean(elem)
                sc_b = elem[baseline_name][sort_metric]
                return sc_m - sc_b
        else:
            def _key_fn(elem):
                return _get_model_mean(elem)
        res_sorted = sorted(res, key=_key_fn, reverse=True)
    elif sort_metric == 'random':
        res_sorted = random.shuffle(res)
    else:
        raise ValueError('`sort_metric` must be one of: {}'.format(METRICS + ['random']))
    return res_sorted


def _display_captions(captions_list, sort_metric):
    # Display captions
    print('')
    instructions = [
        '"x" if both are wrong',
        '"y" if both are correct',
        '"b" if baseline is correct',
        '"m" if model is correct',
        '"a" if ambiguous',
        '"e" to exit',
        'other keys to skip.\n',
    ]
    instructions = '\n'.join(instructions)
    global jump_to_idx
    if jump_to_idx < 0 or jump_to_idx >= len(captions_list):
        jump_to_idx = 0
    img_plot = None
    fig = plt.figure(figsize=(20, 10))
    for cap_idx, cap in enumerate(captions_list[jump_to_idx:]):
        if len(shortlisted_imgs) > 0 and any(str(cap['image_id']) in _ for _ in shortlisted_imgs):
            # If there are shortlisted images, might skip
            pass
        else:
            continue
        img = Image.open(pjoin(image_dir, cap['image_name']))
        img = ImageEnhance.Brightness(img).enhance(1.10)
        img = ImageEnhance.Contrast(img).enhance(1.050)

        # Resize to 512 x 512 instead of 256 x 256
        # Crop to 448 x 448 instead of 224 x 224
        img = img.resize([IMG_RESIZE, IMG_RESIZE], Image.BILINEAR)
        img = ImageOps.crop(img, (IMG_RESIZE - IMG_CROP) / 2)

        # Collect info
        base_score = cap[baseline_name][sort_metric]
        model_score = [cap[n][sort_metric] for n in model_names]
        base_cap = '{} ({:.2f}): {}'.format(
            baseline_name, base_score, cap[baseline_name]['caption'])
        model_cap = ['{} ({:.2f}): {}'.format(
            n, model_score[i], cap[n]['caption']) for i, n in enumerate(model_names)]

        # Visualise
        border = int((IMG_RESIZE - IMG_CROP) / 2)
        bg_big = Image.new('RGB', DISPLAY_BG_SIZE)
        bg_big.paste(img, (border, border))
        draw = ImageDraw.Draw(bg_big)
        draw.text((IMG_RESIZE, border),
                  '# {} / {}'.format(jump_to_idx + cap_idx + 1, len(captions_list)), font=font)

        # Draw captions
        texts_wrp = []
        for t in [base_cap] + model_cap:
            texts_wrp.append(textwrap.wrap(t, width=50))
        offset = 0.
        for text_group in texts_wrp:
            for text in text_group:
                draw.text((border, IMG_RESIZE + offset), text, font=font)
                offset += int(TEXT_SIZE * 1.05)
            offset += TEXT_SIZE
        # draw.text((10, int(IMG_RESIZE * 1.00)), base_cap, font=font)
        # for j, m_cap in enumerate(model_cap):
        #     offset = 0.2 * (j + 1)
        #     draw.text((10, int(IMG_RESIZE * (1. + offset))), m_cap, font=font)

        if img_plot is None:
            img_plot = plt.imshow(bg_big)
        else:
            img_plot.set_data(bg_big)
        plt.show(block=False)
        fig.canvas.draw()

        # Get key press
        # key_input = raw_input(instructions)
        key_input = input(instructions)
        fig.canvas.flush_events()

        if key_input == 'e':
            plt.close()
            break
        elif key_input in CATEGORIES:
            _save_files(CATEGORIES[key_input], img, cap, bg_big, sort_metric)
        print('')


def _save_files(caption_type, img, caption, composite, sort_metric):
    img_id = caption['image_id']
    base_score = caption[baseline_name][sort_metric]
    model_score = [caption[n][sort_metric] for n in model_names]
    base_out = '{} ({}): {}'.format(
        baseline_name, base_score, caption[baseline_name]['caption'])
    model_out = ['{} ({}): {}'.format(
        n, model_score[i], caption[n]['caption']) for i, n in enumerate(model_names)]
    # Save image
    score = '{:1.3f}'.format(model_score[-1]).replace('.', '-')
    type_short = {v: k for k, v in CATEGORIES.items()}
    if type(img_id) == str:
        img_out_name = '{}_{}_{}.jpg'.format(type_short[caption_type], score, img_id)
    else:
        img_out_name = '{}_{}_{:012d}.jpg'.format(type_short[caption_type], score, img_id)
    img.save(pjoin(output_dir, img_out_name))

    draw = ImageDraw.Draw(composite)
    offset = int(IMG_RESIZE - TEXT_SIZE) / 2
    draw.text((IMG_RESIZE, offset), img_out_name, font=font)
    draw.text((IMG_RESIZE, offset + TEXT_SIZE), 'Type: ' + caption_type, font=font)
    composite.save(pjoin(output_dir, 'comp_' + img_out_name))

    # Write captions
    out_str = '{}\r\n{}\r\n\r\n'.format(base_out, '\r\n'.join(model_out))
    with open(pjoin(output_dir, 'captions_{}.txt'.format(caption_type)), 'a') as f:
        f.write('{}\r\n{}'.format(img_out_name, out_str))

    # Write captions in LATEX format
    modcap = '        \\begin{{modcap}}\n'
    modcap += '            {}\n'
    modcap += '        \\end{{modcap}} \\\\\n'
    out_str = [
        '    \\gph{{1.0}}{{resources/xxx/{}}}  &'.format(img_out_name),
        '    \\begin{tabular}{M{\\linewidth}}',
        '        \\begin{basecap}',
        '            {}'.format(caption[baseline_name]['caption']),
        '        \\end{basecap} \\\\',
    ]
    for n in model_names:
        out_str += [modcap.format(caption[n]['caption'])]
    out_str += [
        '    \\end{tabular} &',
        '    ',
    ]

    with open(pjoin(output_dir, 'captions_latex_{}.txt'.format(caption_type)), 'a') as f:
        f.write('\n'.join(out_str) + '\n')


def main():
    os.makedirs(output_dir, exist_ok=True)
    id_to_results = {}

    config = dict(
        sort_by_metric=sort_by_metric,
        baseline_json=baseline_json,
        model_json=model_json,
    )
    with open(pjoin(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Load captions
    for j, n in zip(model_json, model_names):
        _load_caption_json(id_to_results, j, n)
    _load_caption_json(id_to_results, baseline_json, baseline_name)

    # Load scores
    for j, n in zip(model_scores_json, model_names):
        _load_score_json(id_to_results, j, n)
    _load_score_json(id_to_results, baseline_scores_json, baseline_name)

    # Sort captions
    caption_list = _sort_captions(id_to_results,
                                  sort_metric=sort_by_metric,
                                  sort_model=model_names,
                                  use_diff=True)
    _display_captions(caption_list, sort_by_metric)


if __name__ == '__main__':
    main()
