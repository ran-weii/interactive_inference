import time
import pandas as pd

def train(model, train_loader, test_loader, epochs, callback=None, verbose=1):
    history = []
    start_time = time.time()
    for e in range(epochs):
        train_stats = model.run_epoch(train_loader, train=True)
        test_stats = model.run_epoch(test_loader, train=False)
        
        tnow = time.time() - start_time
        train_stats.update({"epoch": e, "time": tnow})
        test_stats.update({"epoch": e, "time": tnow})
        history.append(train_stats)
        history.append(test_stats)

        if (e + 1) % verbose == 0:
            s = model.stdout(train_stats, test_stats)
            print("e: {}/{}, {}, t: {:.2f}".format(e + 1, epochs, s, tnow))

        if callback is not None:
            callback(model, history)

    df_history = pd.DataFrame(history)
    return model, df_history