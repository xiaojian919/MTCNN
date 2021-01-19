import nets
import nets_2
import train
import train_2
import os
if __name__ == '__main__':
    net = nets_2.Pnet()
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    trainer = train_2.Trainer(net, r'.\temp\p_net.pth', r".\CelebA\12",r".\CelebAVal\12")
    trainer.trainer()
