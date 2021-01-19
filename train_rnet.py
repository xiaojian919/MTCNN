import nets
import nets_2
import train
import train_2
import os
if __name__ == '__main__':
    net = nets_2.Rnet()
    if not os.path.exists("./temp"):
        os.makedirs("./temp")
    trainer = train_2.Trainer(net,r".\temp\r_net.pth", r".\CelebA\24",r".\CelebAVal\24")
    trainer.trainer()
