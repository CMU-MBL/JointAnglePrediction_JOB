# -------------------------
#
# Functions to run model training
#
# --------------------------

import torch
import sys
from tqdm import tqdm


def run_stage(model, scheduler, criterion, train_dloader, val_dloader, test_dloader, logger, stage, show_bar=True):
    train_dloader_iter = iter(train_dloader)
    x_val, y_val = next(iter(val_dloader))
    x_test, y_test = next(iter(test_dloader))

    # Shows progress through training iterations
    if show_bar is True:
        if stage['disprog']:
            print('|{0:>3.0f}% |'.format(0), end='')
        with tqdm(total=stage['num_iter'], leave=True, disable=stage['disprog']) as prog_bar:
            # Run through training iterations
            for i in range(0, stage['num_iter']):
                try:
                    x, y = next(train_dloader_iter)
                except StopIteration:
                    train_dloader_iter = iter(train_dloader)
                    x, y = next(train_dloader_iter)

                # Check for validation step
                if (i == 0) or (((i+1) % logger.log_freq) == 0) or ((i+1) == stage['num_iter']):
                    val_loss, y_val_pred = validate(model, criterion, x_val, y_val)

                # Forward pass
                y_pred = model(x)
                loss = criterion(y_pred, y)
                # Check if this is first iteration overall
                if logger.idx == 0:
                    logger.step(loss, val_loss, x_val, y_val_pred, y_val)

                # Backward pass
                loss.backward()
                scheduler.optimizer.step()
                scheduler.optimizer.zero_grad()

                # Log current model
                logger.step(loss, val_loss, x_val, y_val_pred, y_val)

                # Step scheduler
                scheduler.step()

                # Update progress bar
                prog_bar.update(1)
                msg = {'Train loss': '{: <6.4f}'.format(loss),
                       'Val loss': '{: <6.4f}'.format(val_loss)}
                prog_bar.set_postfix(msg)
                prog_bar.refresh()

                if stage['disprog'] and (((i+1) % int(0.05*stage['num_iter'])) == 0):
                    print('{0:>3.0f}% |'.format((i+1)/stage['num_iter']*100), end='')
                    sys.stdout.flush()
    else:
        for i in range(0, stage['num_iter']):
            try:
                x, y = next(train_dloader_iter)
            except StopIteration:
                train_dloader_iter = iter(train_dloader)
                x, y = next(train_dloader_iter)

            # Check for validation step
            if (i == 0) or (((i+1) % logger.log_freq) == 0) or ((i+1) == stage['num_iter']):
                val_loss, y_val_pred = validate(model, criterion, x_val, y_val)

            # Forward pass
            y_pred = model(x)
            loss = criterion(y_pred, y)

            # Check if this is first iteration overall
            if logger.idx == 0:
                logger.step(loss, val_loss, x_val, y_val_pred, y_val)

            # Backward pass
            loss.backward()

            scheduler.optimizer.step()
            scheduler.optimizer.zero_grad()

            # Log current model
            logger.step(loss, val_loss, x_val, y_val_pred, y_val)

            # Step scheduler
            scheduler.step()

    _, y_pred_val = validate(model, criterion, x_val, y_val)
    _, y_pred_test = validate(model, criterion, x_test, y_test)

    print('')
    logger.log_end_of_stage(x_val, y_val, y_pred_val, x_test, y_test, y_pred_test)

    return model, scheduler, logger


def validate(model, criterion, x_val, y_val):
    # Uses final model to predict and return values and loss

    with torch.no_grad():
        model.eval()
        y_pred_val = model(x_val)
        val_loss = criterion(y_pred_val, y_val)
        model.train()

    return (val_loss, y_pred_val)
