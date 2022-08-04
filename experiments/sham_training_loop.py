from core import training as ctr

def main():

    trainer = ctr.sham_trainer()
    trainer.train(10)

if __name__=="__main__":
    main()
