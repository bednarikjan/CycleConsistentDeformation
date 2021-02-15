# import sys
# sys.path.append("./auxiliary/")
# sys.path.append("./extension/")
import cyccon.auxiliary.argument_parser as argument_parser
import cyccon.auxiliary.my_utils as my_utils

opt = argument_parser.parser()
my_utils.plant_seeds(randomized_seed=False)

import cyccon.training.trainer as trainer

trainer = trainer.Trainer(opt)
trainer.build_dataset_train()
trainer.build_network()
trainer.build_optimizer()
trainer.build_losses()

# for epoch in range(opt.nepoch):
while trainer.total_iters < opt.n_train_iters:
    trainer.train_epoch()
    # if epoch % 50 == 0:
    if trainer.total_iters % 2000 == 0:
        trainer.dump_stats()
        trainer.save_network()
    trainer.increment_epoch()
trainer.dump_stats()
trainer.save_network()

trainer.save_new_experiments_results()
