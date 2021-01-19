import nets
import nets_2
import train
import train_2
import os
if __name__ == '__main__':
    net = nets_2.Onet()
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    trainer = train_2.Trainer(net, r'.\temp\o_net.pth', r".\CelebA\48",r".\CelebAVal\48")
    trainer.trainer()