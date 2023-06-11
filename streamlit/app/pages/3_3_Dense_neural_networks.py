import streamlit as st
import os

image_dir = os.path.join(os.path.dirname(__file__), "../images/dense_neural_networks")

st.header("""Dense neural networks""")
st.markdown(
    """
    We got now some basics to finally tackle what is meant with :red[dense neural networks].

    Neural networks are appearing today almost everywhere. You can use them to 
    - classify objects/pixels
    - For regression tasks
    - estimate parameters
    - ...

    But what is the core idea of neural networks?

    Neural Networks (NN) are used to mimic the human brain.

    The core idea is quite simple, by making use of something what is called :red[emergence].

    Emergence means that the interaction of quite simple systems can form a complex systems with quite a lot of capacities
    (think about a population of ants, which for there own are not that impressive, but considering the whole colony it is really
    cool to see what they are capable off :ant:)

    Ok enough of praising ants.

    Let's get to see what the building blocks of so called :red[neural networks] actually are.
    """)

#############################
st.subheader("Neuron")
st.markdown(
    """
    A neuron is probably known to all of us. The human brain consists of around 100 billion neurons. And they all interact by synapsis an electrical and biochemical way.

    A single neuron shows a nonlinear input/output behavior, it *fires* only if a specific excitation threshold is reached.

    Now how does an artificial neuron look like?
    """)

st.image(os.path.join(image_dir, "artificial_neuron.png"),
         caption="A biological and an artificial neuron (via https://www.quora.com/What-is-the-differences-between-artificial-neural-network-computer-science-and-biological-neural-network)", width=600)

st.markdown("""Ok but what does this mean? :brain:""")
st.markdown("""
Easy. A single neuron is a small system, which takes inputs, sums them up and creates and output.

Not to believe that such a system is capable of doing that fascinating stuff? It does only some linear transformation, by multiplying :red[inputs] with something which is called :red[weight], adds everything
together and also adds a so called :red[bias].

But yes, as you probably already have assumed that is not the magic of neural networks.

Performing linear transformations after each other is not that special. But here comes the trick.

Neural networks also include something which is called :red[activation function]. And that's where it get's interesting.

Those activation functions are chained after the neuron to perform some non-linear mapping. And now think of chaining a lot of non-linear mappings
where the individual weights and biases are coming with huge flexibility.

Indeed, by doing so you are able to approximate every function you want!
""")

st.markdown("---")
############################
st.subheader("Layer")
st.markdown("---")
############################
st.subheader("Feed forward neural network")
st.markdown("---")
############################
st.subheader("Activation function")
st.markdown("---")
############################
st.subheader("Universal approximation")
st.markdown("---")
############################
st.subheader("Loss and cost function")
st.markdown("---")
############################
st.subheader("Training")
st.markdown("---")
############################
st.subheader("Implementation")
st.markdown("---")
############################
