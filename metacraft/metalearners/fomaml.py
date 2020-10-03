from metacraft.metalearners.maml import MAML

class FOMAML(MAML):
    def __init__(self, model, outer_optimizer, loss_function, inner_lr,
                 inner_steps = 1, device = None):
        
        super(FOMAML, self).__init__(
            model = model,
            outer_optimizer = outer_optimizer,
            loss_function = loss_function,
            inner_lr = inner_lr,
            inner_steps = inner_steps,
            first_order = True,
            device = device
        )