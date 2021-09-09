from utils import *
import tqdm
import argparse
import multiprocessing as mp


def build_argparser():
    paser = argparse.ArgumentParser()
    paser.add_argument('--video', type=str, required=True)
    # paser.add_argument('--size', type=int, required=True)
    paser.add_argument('--scale', type=int, default=40)
    paser.add_argument('--workers', type=int, default=1)
    paser.add_argument('--mapping', type=str, default='mapping.json')
    return paser


def patch_preprocess(patch, img, mapping, inverse_mapping, opts):
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            sub_img = read_img(mapping[find_closest(img[r, c], inverse_mapping)])
            patch[r * opts.scale:(r + 1) * opts.scale, c * opts.scale:(c + 1) * opts.scale, :] = cv2.resize(sub_img,
                                                                                                            (opts.scale,
                                                                                                             opts.scale))


def main():
    parser = build_argparser()
    opts = parser.parse_args()
    directory = 'tmp'
    size = (1080, 1920)

    mapping_from_video(opts.video, opts.mapping)

    with open('test.json', 'r') as f:
        mappings = json.load(f)
    inverse_mapping = []
    for key in mappings.keys():
        inverse_mapping.append([int(x) for x in key.split('-')])
    inverse_mapping = np.array(inverse_mapping)

    videoWriter = cv2.VideoWriter('out.mp4', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 24, size)

    imgs = os.listdir(directory)
    cur = 0
    for img in tqdm.tqdm(imgs):
        img = read_img(os.path.join(directory, img))
        res = np.zeros_like(img)
        img = cv2.resize(img, (img.shape[1] // opts.scale, img.shape[0] // opts.scale))
        pool = [mp.Process(target=patch_preprocess,
                           args=(res[i * res.shape[0] // opts.workers:(i + 1) * res.shape[0] // opts.workers, :, :],
                                 img[i * img.shape[0] // opts.workers:(i + 1) * img.shape[0] // opts.workers, :, :],
                                 mappings, inverse_mapping, opts))
                for i in range(opts.workers)]
        for p in pool:
            p.start()
        for p in pool:
            p.join()
        videoWriter.write(res)
    videoWriter.release()


if __name__ == '__main__':
    main()
