.. _my-customalgorithm:



Custom algorithm
===========================

Surrogate Gradient Algorithms
--------------------------------------
The most import part of surrogate gradient algorithms is that use custom gradient function to replace the original \
backpropagation gradient. Here we use **STCA** and **STBP** as example to show how to use custom gradient formula.

In **STCA** [#f1]_ learning algorithm, the graident function is:
:math:`h(V)=\frac{1}{\alpha}sign(|V-\theta|<\alpha)`

In **STBP** [#f2]_ learning algorithm, the graident function is:
:math:`h_4(V)=\frac{1}{\sqrt{2\pi a_4}} e^{-\frac{(V-V_th)^2)}{2a_4}}`



.. code-block:: python

    @staticmethod
    def backward(
            ctx,
            grad_output
        ):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input - ctx.thresh) < ctx.alpha  # According to STCA learning algorithm
        # temp = torch.exp(-(input - ctx.thresh) ** 2 / (2 * ctx.alpha)) \  # According to STBP learning algorithm
        #                  / (2 * math.pi * ctx.alpha)
        result = grad_input * temp.float()
        return result, None, None


Synaptic Plasticity Algorithms
---------------------------------
We have constructed two kinds of STDP learning algorithm. The first one is based on the global synaptic plasticity, we call it ``full_online_STDP`` [#f3]_ ,\
another one is based on the nearest synaptic plasticity, we call it ``nearest_online_STDP`` [#f4]_ .

Full Synaptic Plasticity STDP learning algorithm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The weight update formula of this algorithm [#f2]_ :
:math:`dw = Apost * (output_spike * input_trace) – Apre * (output_trace * input_spike)`
:math:`weight = weight + dw`
Weight normalization formula:
:math:`weight = self.w_norm * weight/sum(torch.abs(weight))`

At first, get the presynaptic and postsynaptic NeuronGroups from :code:`trainable_connection` :

.. code-block:: python

    preg = conn.pre_assembly
    postg = conn.post_assembly

Then, get parameters ID, such as input spike, output spike and weight name:

.. code-block:: python

    pre_name = conn.get_input_name(preg, postg)
    post_name = conn.get_group_name(postg, 'O')
    weight_name = conn.get_link_name(preg, postg, 'weight')

Add necessary parameters to ``Backend`` :

.. code-block:: python

    backend.add_variable(input_trace_name, backend._variables[pre_name].shape, value=0.0)
    backend.add_variable(output_trace_name, backend._variables[post_name].shape, value=0.0)
    backend.add_variable(dw_name, backend._variables[weight_name].shape, value=0.0)

Append calculate formula to ``Backend`` :

.. code-block:: python

    backend.add_operation(['input_trace_temp', 'var_mult', input_trace_name, 'trace_decay'])
    backend.add_operation([input_trace_name, 'add', pre_name, 'input_trace_temp'])

    backend.add_operation(['output_trace_temp', 'var_mult', output_trace_name, 'trace_decay'])
    backend.add_operation([output_trace_name, 'add', post_name, 'output_trace_temp'])

    backend.add_operation(['pre_post_temp', 'mat_mult_pre', post_name, input_trace_name+'[updated]'])
    backend.add_operation(['pre_post', 'var_mult', 'Apost', 'pre_post_temp'])
    backend.add_operation(['post_pre_temp', 'mat_mult_pre', output_trace_name+'[updated]', pre_name])
    backend.add_operation(['post_pre', 'var_mult', 'Apre', 'post_pre_temp'])
    backend.add_operation([dw_name, 'minus', 'pre_post', 'post_pre'])
    backend.add_operation([weight_name, self.full_online_stdp_weightupdate, dw_name, weight_name])

Weight update part:

.. code-block:: python

    with torch.no_grad():
        weight.add_(dw)

Weight normalization part:

.. code-block:: python

    weight[...] = (self.w_norm * torch.div(weight, torch.sum(torch.abs(weight), 1, keepdim=True)))
    weight.clamp_(0.0, 1.0)


.. [#f1]  Pengjie Gu et al. "STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep SpikingNeural Networks." In:Proceedings of the Twenty-Eighth International Joint Conference on Artificial Intelligence, IJCAI-19. International Joint Conferences on Artificial Intelligence Organization, July 2019,pp. 1366–1372. `doi:10.24963/ijcai.2019/189. <https://doi.org/10.24963/ijcai.2019/189>`_
.. [#f2]  Yujie Wu et al. "Spatio-Temporal Backpropagation for Training High-Performance Spiking Neural Networks" Front. Neurosci., 23 May 2018 | `doi:10.3389/fnins.2018.00331<https://doi.org/10.3389/fnins.2018.00331>`_
.. [#f3]  Sjöström J, Gerstner W. Spike-timing dependent plasticity[J]. Spike-timing dependent plasticity, 2010, 35(0): 0-0._
.. [#f4]  Gerstner W, Kempter R, van Hemmen JL, Wagner H. A neuronal learning rule for sub-millisecond temporal coding. Nature. 1996 Sep 5;383(6595):76-81. `doi: 10.1038/383076a0<https://doi.org/10.1038/383076a0>`_ . PMID: 8779718.

