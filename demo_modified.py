# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test_modified import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0

    FPS = 24
    SIZE = (854, 480)
    PATH = './drone_modified.avi'
    video = cv2.VideoWriter(PATH, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), FPS, SIZE)
    im_tmp = ims[0]
    cv2.rectangle(im_tmp, (x, y), (x + w, y + h), (255, 0, 0), 5)
    for i in range(24): video.write(im_tmp)

    #new variable
    a, b, c = 0.5, 0.8, 0.995
    prob_lost = False
    bel_0 = 1
    bel_t = bel_0
    n_0, n_t = 0, 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
            state_init = state
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask'] > state['p'].seg_thr
            # im[:, :, 0] = (state['mask'] < 0) * 255 + (state['mask'] >= 0) * im[:, :, 0]
            im[:, :, 2] = (mask > 0) * 255 + (mask == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            cv2.waitKey(1)
            n_0 = np.count_nonzero(np.asarray(state['mask'] > state['p'].seg_thr))
            print('n_0: ', n_0)
        elif f > 0:  # tracking
            state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            location = state['ploygon'].flatten()
            mask = state['mask']
            mask_bin = mask > state['p'].seg_thr

            # new strategy
            bel_0 = c * bel_0
            n_t = np.count_nonzero(np.asarray(mask_bin))
            if prob_lost == False and n_t/n_0 < a:
                prob_lost = True
                print('lost in frame {}'.format(f))
            elif prob_lost == True and n_t/n_0 > b:
                prob_lost = False
            if prob_lost == True:
                bel_t = n_t / (n_0 * a)
                print('bel_0: {}, bel_t: {}'.format(bel_0, bel_t))
            state_0 = siamese_track(state_init, im, mask_enable=True, refine_enable=True, device=device)  # track
            mask_0 = state_0['mask']
            mask = (mask < 0) * 0 + (mask >= 0) * mask
            mask_0 = (mask_0 < 0) * 0 + (mask_0 >= 0) * mask_0
            mask_final = (bel_t / (bel_t + bel_0)) * mask + (bel_0 / (bel_t + bel_0)) * mask_0
            # mask_final = mask_0
            mask_final_bin = mask_final > state['p'].seg_thr

            # im[:, :, 0] = (mask_final < 0) * 255 + (mask_final >= 0) * im[:, :, 0]
            im[:, :, 2] = (mask_final_bin > 0) * 255 + (mask_final_bin == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            cv2.imshow('SiamMask', im)
            # cv2.imshow('mask', mask)
            # cv2.imshow('mask_0', mask_0)
            key = cv2.waitKey(1)
            if key > 0:
                break
            video.write(im)

        toc += cv2.getTickCount() - tic
    video.release()
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
