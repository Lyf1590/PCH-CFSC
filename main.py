import time
import settings
from train import Session
# from TEST import Session
import torch
import os
import numpy as np
from torch.backends import cudnn

# import xlwt
# from openpyxl import Workbook

def main():
    os.environ['PYTHONHASHSEED'] = str(settings.SEED)
    np.random.seed(settings.SEED)
    torch.manual_seed(settings.SEED)
    torch.cuda.manual_seed(settings.SEED)
    torch.cuda.manual_seed_all(settings.SEED)
    torch.cuda.set_device(settings.GPU)
    torch.backends.cudnn.deterministic = True

    logger = settings.logger
    sess = Session()

    best_it = best_ti = 0



    if sess.COM_PART == True:
        for epoch in range(settings.NUM_EPOCH):
            sess.com_train(epoch)
            if (epoch + 1) % settings.EVAL_INTERVAL == 0:
                MAP_I2T, MAP_T2I = sess.eval()
                if (best_it + best_ti) < (MAP_I2T + MAP_T2I):
                    best_it, best_ti = MAP_I2T, MAP_T2I
                    print('Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (best_it, best_ti))
                    logger.info('Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (best_it, best_ti))
                    sess.save_checkpoints('checkpoint')
        logger.info('Final  Best MAP of I->T: %.4f, Best mAP of T->I: %.4f' % (best_it, best_ti))

    else:
        logger.info('------------------------------------------------------------------------------------')
        sess.partial_train()
        # start2_time = time.time()
        # MAP_I2T, MAP_T2I = sess.eval()
        # logger.info('------------------------------------------------------------------------------------')
        # logger.info(f'Final mAP@50 of I->T: {MAP_I2T:.4f}')
        # logger.info(f'Final mAP@50 of T->I: {MAP_T2I:.4f}')
        # logger.info('------------------------------------------------------------------------------------')
        # end2_time = time.time()
        #
        #
        # train_time = end2_time - start2_time
        # logger.info('---------------- Training time: {:.3f} ---------------------------------'.format(train_time))

        # query_time = end2_time - start2_time
        # logger.info('----------------  Querying time: {:.3f} ---------------------------------'.format(query_time))


if __name__ == '__main__':
    main()
