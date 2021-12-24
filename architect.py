import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable


def _concat(xs):
    return torch.cat([x.view(-1) for x in xs])


class Architect(object):

    def __init__(self, domain_model, general_model, classification_head, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.domain_model = domain_model
        self.general_model = general_model
        self.classification_head = classification_head
        self.optimizer = torch.optim.Adam(self.domain_model.arch_parameters(),
                                          lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)

    def _compute_unrolled_model(self, source_input, source_domain, target_input, target_domain, eta, network_optimizer):
        source_domain_loss = self.domain_model._loss(
            source_input, source_domain)
        target_domain_loss = self.domain_model._loss(
            target_input, target_domain)
        total_loss = source_domain_loss + target_domain_loss
        theta = _concat(self.domain_model.parameters()).data
        try:
            moment = _concat(network_optimizer.state[v]['momentum_buffer']
                             for v in self.domain_model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)
        dtheta = _concat(torch.autograd.grad(
            total_loss, self.domain_model.parameters(), retain_graph=True, allow_unused=True)).data + self.network_weight_decay*theta
        unrolled_model = self._construct_model_from_theta(
            theta.sub(eta, moment+dtheta))
        return unrolled_model

    def _compute_unrolled_model1(self, source_input, source_target, target_input, target_target, eta, unrolled_model, network_optimizer, general_optimizer, head_optimizer, input_dim):
        domain_rep_source, _ = self.domain_model(source_input)
        domain_rep_target, _ = self.domain_model(target_input)
        general_rep_soruce = self.general_model(source_input)
        general_rep_target = self.general_model(target_input)
        class_rep_source = general_rep_soruce - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = self.classification_head(class_rep_source)
        logits_class_target = self.classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss1 = crit(logits_class_source, source_target)
        loss2 = crit(logits_class_target, target_target)
        loss = loss1+loss2
        # source_class_loss = self.classification_head._loss(
        # class_rep_source, source_target)
        # target_class_loss = self.classification_head._loss(
        # class_rep_target, target_target)
        #total_class_loss = source_class_loss+target_class_loss

        theta1 = _concat(self.general_model.parameters()).data
        theta2 = _concat(self.classification_head.parameters()).data
        try:
            moment1 = _concat(general_optimizer.state[v]['momentum_buffer']
                              for v in self.general_model.parameters()).mul_(self.network_momentum)
            moment2 = _concat(head_optimizer.state[v]['momentum_buffer']
                              for v in self.classification_head.parameters()).mul_(self.network_momentum)
        except:
            moment1 = torch.zeros_like(theta1)
            moment2 = torch.zeros_like(theta2)

        dtheta1 = _concat(torch.autograd.grad(
            loss, self.general_model.parameters(), retain_graph=True, allow_unused=True)).data + self.network_weight_decay*theta1
        dtheta2 = _concat(torch.autograd.grad(
            loss, self.classification_head.parameters(), retain_graph=True, allow_unused=True)).data + self.network_weight_decay*theta2
        unrolled_general_model = self._construct_model_from_theta1(
            theta1.sub(eta, moment1+dtheta1), input_dim)
        unrolled_classification_head = self._construct_model_from_theta2(
            theta2.sub(eta, moment2+dtheta2))
        return unrolled_general_model, unrolled_classification_head

    def step(self, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, input_source_valid, target_source_valid, domain_source_valid, input_target_valid, target_target_valid, domain_target_valid,
             eta, network_optimizer, general_optimizer, head_optimizer, input_dim, unrolled):
        self.optimizer.zero_grad()
        if unrolled:
            self._backward_step_unrolled(
                source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, input_source_valid, target_source_valid, domain_source_valid, input_target_valid, target_target_valid, domain_target_valid, eta, network_optimizer, general_optimizer, head_optimizer, input_dim)
        else:
            self._backward_step(input_source_valid, target_source_valid, domain_source_valid,
                                input_target_valid, target_target_valid, domain_target_valid)

        self.optimizer.step()

    def step1(self, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, input_source_valid, target_source_valid, domain_source_valid, input_target_valid, target_target_valid, domain_target_valid,
              eta, network_optimizer, general_optimizer, head_optimizer, input_dim, unrolled):
        self.optimizer.zero_grad()

        unrolled_model = self._compute_unrolled_model(
            source_input_train, source_domain_train, target_input_train, target_domain_train, eta, network_optimizer)
        unrolled_general_model, unrolled_classification_head = self._compute_unrolled_model1(
            source_input_train, source_target_train, target_input_train, target_target_train, eta, unrolled_model, network_optimizer, general_optimizer, head_optimizer, input_dim)

        general_rep_source = unrolled_general_model(input_source_valid)
        general_rep_target = unrolled_general_model(input_target_valid)
        domain_rep_source, _ = self.domain_model(input_source_valid)
        domain_rep_target, _ = self.domain_model(input_target_valid)
        class_rep_source = general_rep_source - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = unrolled_classification_head(class_rep_source)
        logits_class_target = unrolled_classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss1 = crit(logits_class_source, target_source_valid)
        loss2 = crit(logits_class_target, target_target_valid)
        loss = loss1+loss2

        loss.backward(retain_graph=True)

        vector_v_dash = [
            v.grad.data for v in unrolled_general_model.parameters()]

        implicit_grads = self._outer1(
            vector_v_dash, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, self.domain_model, unrolled_general_model, unrolled_classification_head, eta)

        for v, g in zip(self.domain_model.arch_parameters(), implicit_grads):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def step2(self, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, input_source_valid, target_source_valid, domain_source_valid, input_target_valid, target_target_valid, domain_target_valid,
              eta, network_optimizer, general_optimizer, head_optimizer, input_dim, unrolled):
        self.optimizer.zero_grad()

        unrolled_model = self._compute_unrolled_model(
            source_input_train, source_domain_train, target_input_train, target_domain_train, eta, network_optimizer)
        unrolled_general_model, unrolled_classification_head = self._compute_unrolled_model1(
            source_input_train, source_target_train, target_input_train, target_target_train, eta, unrolled_model, network_optimizer, general_optimizer, head_optimizer, input_dim)

        general_rep_source = unrolled_general_model(input_source_valid)
        general_rep_target = unrolled_general_model(input_target_valid)
        domain_rep_source, _ = self.domain_model(input_source_valid)
        domain_rep_target, _ = self.domain_model(input_target_valid)
        class_rep_source = general_rep_source - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = unrolled_classification_head(class_rep_source)
        logits_class_target = unrolled_classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss1 = crit(logits_class_source, target_source_valid)
        loss2 = crit(logits_class_target, target_target_valid)
        loss = loss1+loss2

        loss.backward(retain_graph=True)

        vector_g_dash = [
            v.grad.data for v in unrolled_classification_head.parameters()]

        implicit_grads = self._outer2(
            vector_g_dash, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, self.domain_model, unrolled_general_model, unrolled_classification_head, eta)

        for v, g in zip(self.domain_model.arch_parameters(), implicit_grads):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

        self.optimizer.step()

    def _backward_step(self, input_valid, domain_valid, target_valid):
        '''general_rep = self.class_model(input_valid)
        domain_rep, logits = self.domain_model(input_valid)
        loss = self.domain_model(input_valid, domain_valid)
        loss.backward()
        class_rep = general_rep - (domain_rep*(10**7))
        loss1 = self.class_model._loss(class_rep, target_valid)
        loss1.backward()'''

    def _backward_step_unrolled(self, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, input_source_valid, target_source_valid, domain_source_valid, input_target_valid, target_target_valid, domain_target_valid, eta, network_optimizer, general_optimizer, head_optimizer, input_dim):
        unrolled_model = self._compute_unrolled_model(
            source_input_train, source_domain_train, target_input_train, target_domain_train, eta, network_optimizer)
        unrolled_general_model, unrolled_classification_head = self._compute_unrolled_model1(
            source_input_train, source_target_train, target_input_train, target_target_train, eta, unrolled_model, network_optimizer, general_optimizer, head_optimizer, input_dim)

        general_rep_source = unrolled_general_model(input_source_valid)
        general_rep_target = unrolled_general_model(input_target_valid)
        domain_rep_source, _ = self.domain_model(input_source_valid)
        domain_rep_target, _ = self.domain_model(input_target_valid)
        class_rep_source = general_rep_source - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = unrolled_classification_head(class_rep_source)
        logits_class_target = unrolled_classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss1 = crit(logits_class_source, target_source_valid)
        loss2 = crit(logits_class_target, target_target_valid)
        loss = loss1+loss2

        loss.backward(retain_graph=True)

        dalpha = [v.grad for v in self.domain_model.arch_parameters()]
        vector = [v.grad.data for v in self.domain_model.parameters()]

        implicit_grads = self._hessian_vector_product(
            vector, source_input_train, source_domain_train, target_input_train, target_domain_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        for v, g in zip(self.domain_model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = Variable(g.data)
            else:
                v.grad.data.copy_(g.data)

    def _construct_model_from_theta(self, theta):
        model_new = self.domain_model.new()
        model_dict = self.domain_model.state_dict()

        params, offset = {}, 0
        for k, v in self.domain_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _construct_model_from_theta1(self, theta, input_dim):
        model_new = self.general_model.new(input_dim)
        model_dict = self.general_model.state_dict()

        params, offset = {}, 0
        for k, v in self.general_model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _construct_model_from_theta2(self, theta):
        model_new = self.classification_head.new()
        model_dict = self.classification_head.state_dict()

        params, offset = {}, 0
        for k, v in self.classification_head.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        model_dict.update(params)
        model_new.load_state_dict(model_dict)
        return model_new.cuda()

    def _hessian_vector_product(self, vector, source_input_train, source_domain_train, target_input_train, target_domain_train, r=1e-2):
        R = r / _concat(vector).norm()

        for p, v in zip(self.domain_model.parameters(), vector):
            p.data.add_(R, v)
        source_loss = self.domain_model._loss(
            source_input_train, source_domain_train)
        target_loss = self.domain_model._loss(
            target_input_train, target_domain_train)
        total_loss = source_loss + target_loss
        grads_p1 = torch.autograd.grad(
            total_loss, self.domain_model.arch_parameters())

        for p, v in zip(self.domain_model.parameters(), vector):
            p.data.sub_(2*R, v)
        source_loss = self.domain_model._loss(
            source_input_train, source_domain_train)
        target_loss = self.domain_model._loss(
            target_input_train, target_domain_train)
        total_loss = source_loss + target_loss
        grads_p2 = torch.autograd.grad(
            total_loss, self.domain_model.arch_parameters())

        for p, v in zip(self.domain_model.parameters(), vector):
            p.data.add_(R, v)

        return [(x-y).div_(2*R) for x, y in zip(grads_p1, grads_p2)]

    def _outer1(self, vector_v_dash, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, unrolled_model, unrolled_general_model, unrolled_classification_head, eta, r=1e-2):
        R1 = r / _concat(vector_v_dash).norm()

        for p, v in zip(self.general_model.parameters(), vector_v_dash):
            p.data.add_(R1, v)

        general_rep_source = self.general_model(source_input_train)
        general_rep_target = self.general_model(target_input_train)
        domain_rep_source, _ = unrolled_model(source_input_train)
        domain_rep_target, _ = unrolled_model(target_input_train)
        class_rep_source = general_rep_source - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target

        logits_class_source = self.classification_head(class_rep_source)
        logits_class_target = self.classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss11 = crit(logits_class_source, source_target_train)
        loss21 = crit(logits_class_target, target_target_train)
        loss1 = loss11+loss21

        loss1.backward(retain_graph=True)
        vector_w_dash = [
            v.grad.data for v in unrolled_model.parameters()]

        grad_part1 = self._hessian_vector_product(
            vector_w_dash, source_input_train, source_domain_train, target_input_train, target_domain_train)

        for p, v in zip(self.general_model.parameters(), vector_v_dash):
            p.data.sub_(2*R1, v)
        general_rep_source = self.general_model(source_input_train)
        general_rep_target = self.general_model(target_input_train)
        domain_rep_source, _ = unrolled_model(source_input_train)
        domain_rep_target, _ = unrolled_model(target_input_train)
        class_rep_source = general_rep_source - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = self.classification_head(class_rep_source)
        logits_class_target = self.classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss12 = crit(logits_class_source, source_target_train)
        loss22 = crit(logits_class_target, target_target_train)
        loss2 = loss12+loss22

        loss2.backward(retain_graph=True)
        vector_w_dash = [
            v.grad.data for v in unrolled_model.parameters()]

        grad_part2 = self._hessian_vector_product(
            vector_w_dash, source_input_train, source_domain_train, target_input_train, target_domain_train)

        for p, v in zip(self.general_model.parameters(), vector_v_dash):
            p.data.add_(R1, v)

        return [(x-y).div_((2*R1)/eta*eta) for x, y in zip(grad_part1, grad_part2)]

    def _outer2(self, vector_v_dash, source_input_train, source_target_train, source_domain_train, target_input_train, target_target_train, target_domain_train, unrolled_model, unrolled_general_model, unrolled_classification_head, eta, r=1e-2):
        R1 = r / _concat(vector_v_dash).norm()

        for p, v in zip(self.classification_head.parameters(), vector_v_dash):
            p.data.add_(R1, v)

        general_rep_source = self.general_model(source_input_train)
        general_rep_target = self.general_model(target_input_train)
        domain_rep_source, _ = unrolled_model(source_input_train)
        domain_rep_target, _ = unrolled_model(target_input_train)
        class_rep_source = general_rep_source - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = self.classification_head(class_rep_source)
        logits_class_target = self.classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss11 = crit(logits_class_source, source_target_train)
        loss21 = crit(logits_class_target, target_target_train)
        loss1 = loss11+loss21

        loss1.backward(retain_graph=True)
        vector_w_dash = [
            v.grad.data for v in unrolled_model.parameters()]

        grad_part1 = self._hessian_vector_product(
            vector_w_dash, source_input_train, source_domain_train, target_input_train, target_domain_train)

        for p, v in zip(self.classification_head.parameters(), vector_v_dash):
            p.data.sub_(2*R1, v)
        general_rep_source = self.general_model(source_input_train)
        general_rep_target = self.general_model(target_input_train)
        domain_rep_source, _ = unrolled_model(source_input_train)
        domain_rep_target, _ = unrolled_model(target_input_train)
        class_rep_source = general_rep_source - domain_rep_source
        class_rep_target = general_rep_target - domain_rep_target
        logits_class_source = self.classification_head(class_rep_source)
        logits_class_target = self.classification_head(class_rep_target)

        crit = nn.CrossEntropyLoss()
        crit = crit.cuda()
        loss12 = crit(logits_class_source, source_target_train)
        loss22 = crit(logits_class_target, target_target_train)
        loss2 = loss12+loss22

        loss2.backward(retain_graph=True)
        vector_w_dash = [
            v.grad.data for v in unrolled_model.parameters()]

        grad_part2 = self._hessian_vector_product(
            vector_w_dash, source_input_train, source_domain_train, target_input_train, target_domain_train)

        for p, v in zip(self.classification_head.parameters(), vector_v_dash):
            p.data.add_(R1, v)

        return [(x-y).div_((2*R1)/eta*eta) for x, y in zip(grad_part1, grad_part2)]
