import os
import io
import glob
import matplotlib.pyplot as plt
import imageio

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SIZE_GUIDANCE = {
    'images': 20
}


def plot_alignment(log_dir, save_dir):
    tf_event = glob.glob(os.path.join(log_dir, "events.*"))[0]
    event_acc = EventAccumulator(tf_event, size_guidance=SIZE_GUIDANCE)
    event_acc.Reload()
    alignments = event_acc.Images('alignment')
    for alignment in alignments:
        img_str = alignment.encoded_image_string
        step = alignment.step
        f = io.BytesIO(img_str)
        img = plt.imread(f)
        plt.imshow(img)
        plt.axis('off')
        plt.title('step {:0>5d}'.format(step))
        plt.tight_layout()
        plt.savefig('{}/{:0>5d}.png'.format(save_dir, step))
        plt.clf()


def make_alignment_gif(log_dir):
    save_dir = log_dir.split(os.sep)[-2] + '-alignmets'
    gif_fp = os.path.join(save_dir, 'alignments.gif')
    os.makedirs(save_dir, exist_ok=True)
    plot_alignment(log_dir, save_dir)

    png_fns = sorted([fn for fn in os.listdir(save_dir) if fn.endswith('.png')])
    images = []
    for fn in png_fns:
        png_fp = os.path.join(save_dir, fn)
        images.append(imageio.imread(png_fp))
    imageio.mimsave(gif_fp, images, duration=0.5)


if __name__ == '__main__':
    log_dir1 = 'NV-tacotron2-log'
    make_alignment_gif(log_dir1)