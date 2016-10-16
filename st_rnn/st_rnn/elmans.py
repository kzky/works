"""Elamn Models
"""

def forward_backward_update_050(
        rnn,
        rnn_labeled_loss,
        rnn_unlabeled_loss,
        optimizer,
        model, 
        x_l, y_l, x_u):

    T = rnn.T

    # Forward/Backward/Update in labeled graph
    model.reset_states()
    model.cleargrads()
    x_list = [x_l for _ in range(T)]
    y_list = rnn(x_list)
    l_losses = rnn_labeled_loss(y_list, y_l)
    loss = reduce(lambda x, y: x + y, l_losses)
    loss.backward()
    optimizer.update()
        
    # Forward/Backward/Update in unlabeled graph
    model.reset_states()
    model.cleargrads()
    x_list = [x_l for _ in range(T)]
    y_list = rnn(x_list)
    u_losses = rnn_unlabeled_loss(y_list)
    loss = reduce(lambda x, y: x + y, u_losses)
    loss.backward()
    optimizer.update()
    
def evaluate_050(
        rnn,
        rnn_labeled_loss,
        x_l, y_l):

    T = rnn.T

    # Forward/Backward/Update in labeled graph
    model.reset_states()
    x_list = [x_l for _ in range(T)]
    y_list = rnn(x_list)
    l_losses = rnn_labeled_loss(y_list, y_l)
    return l_losses

        
    
