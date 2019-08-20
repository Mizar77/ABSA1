import os
import sys
import torch
import torch.autograd as autograd
import torch.nn.functional as F


def train(train_iter, dev_iter, model, args, model_name):
    if args.cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    steps = 0
    best_acc = 0
    last_step = 0
    model.train()
    # 训练模式，主要是针对model 在训练时和评价时不同的 Batch Normalization  和  Dropout 方法模式
    for epoch in range(1, args.epochs+1):
        for batch in train_iter:
            feature, aspect, target = batch.text, batch.aspect, batch.label
            # feature.data.t_(), target.data.sub_(1)  # batch first, index align
            # print(feature)
            feature = feature.data.t()
            aspect = aspect.data.t()
            target = target.data.sub(1)
            # print(feature)
            if args.cuda:
                feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

            optimizer.zero_grad()
            logit = model(feature, aspect)

            # print('logit vector', logit.size(), logit)
            # print('target vector', target.size(), target)

            loss = F.cross_entropy(logit, target)
            loss.backward()
            optimizer.step()

            steps += 1
            if steps % args.log_interval == 0:
                corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
                accuracy = 100.0 * corrects/batch.batch_size
                sys.stdout.write(
                    '\r{}-Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(model_name,
                                                                                steps,
                                                                             # loss.data[0],
                                                                             loss.item(),
                                                                             accuracy,
                                                                             corrects,
                                                                             batch.batch_size))
            if steps % args.test_interval == 0:
                dev_acc = eval(dev_iter, model, args)
                if dev_acc > best_acc:
                    best_acc = dev_acc
                    last_step = steps
                    if args.save_best:
                        save(model, args.save_dir, 'best', steps)
                else:
                    if steps - last_step >= args.early_stop:
                        print('early stop by {} steps.'.format(args.early_stop))
            elif steps % args.save_interval == 0:
                save(model, args.save_dir, 'snapshot', steps)


def eval(data_iter, model, args):
    model.eval()
    # 测试模式，eval（）时，pytorch会自动把BN（batch-normalization)和DropOut固定住，不会取平均，而是用训练好的值
    corrects, avg_loss = 0, 0
    print()
    for batch in data_iter:
        feature, aspect, target = batch.text, batch.aspect, batch.label
        feature = feature.data.t()
        aspect = aspect.data.t()
        target = target.data.sub(1)
        if args.cuda:
            feature, aspect, target = feature.cuda(), aspect.cuda(), target.cuda()

        logit = model(feature, aspect)
        loss = F.cross_entropy(logit, target, size_average=False)
        # 与criteria = torch.nn.CrossEntropyLoss()
        # loss=criteria(logit,target)功能相同

        # avg_loss += loss.data[0]
        avg_loss += loss.item()
        corrects += (torch.max(logit, 1)
                     [1].view(target.size()).data == target.data).sum()

    size = len(data_iter.dataset)
    avg_loss /= size
    accuracy = 100.0 * corrects/size
    print('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                       size))
    return accuracy


def predict(text, model, text_field, label_feild, cuda_flag):
    assert isinstance(text, str)
    model.eval()
    # text = text_field.tokenize(text)
    text = text_field.preprocess(text)
    text = [[text_field.vocab.stoi[x] for x in text]]
    x = torch.tensor(text)
    x = autograd.Variable(x)
    if cuda_flag:
        x = x.cuda()
    print(x)
    output = model(x)
    _, predicted = torch.max(output, 1)
    # return label_feild.vocab.itos[predicted.data[0][0]+1]
    return label_feild.vocab.itos[predicted.data[0]+1]


def save(model, model_name, save_dir, save_prefix, steps):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_prefix = os.path.join(save_dir, save_prefix)
    save_path = '{}_{}_steps_{}.pt'.format(save_prefix, model_name, steps)
    torch.save(model.state_dict(), save_path)
