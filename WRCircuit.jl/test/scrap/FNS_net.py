if __name__ == "__main__":
    plt.style.use("foresight.mplstyle")
    num_exc_neurons = (100, 100)
    num_inh_neurons = 100
    FNSnet = WRCircuit(num_exc_neurons, num_inh_neurons)

    input, T = bp.inputs.section_input([0.0, 400.0], [100.0, 100.0], return_length=True)
    inputs = np.zeros(FNSnet.E.size + (len(input),))
    inputs[0, 0] = input
    inputs = inputs.reshape(-1, inputs.shape[-1])

    inputs = np.zeros(FNSnet.E.size).flatten() + 400.0
    inputs = np.tile(
        inputs, (int(T / bp.share.load("dt")), 1)
    ).transpose()  # A matrix doesn't work. Maybe add a new input variable post-hoc that targets a specific set of indices?
    print(inputs.shape)

    runner = bp.DSRunner(
        FNSnet,
        monitors=["E.spike", "I.spike"],
        inputs=[
            ("Ein.input", inputs),
            ("Iin.input", 0.0),
        ],
    )
    runner.run(T)
    t = runner.mon["ts"].view()  # [1000:]
    X = runner.mon["E.spike"].view()  # [1000:]

    if False:
        conn = FNSnet.E2E.proj.comm.conn
        positions = FNSnet.E.positions
        if len(positions[0]) == 1:
            positions = [(x[0], 0) for x in positions]
        G = nx.from_numpy_array(conn.require("conn_mat"), create_using=nx.DiGraph())
        nx.draw(
            G,
            pos=positions,
            connectionstyle="arc3,rad=0.5",
            node_color="b",
            edge_color="b",
            node_size=20,
            width=0.5,
        )

        conn = FNSnet.I2I.proj.comm.conn
        positions = FNSnet.I.positions
        if len(positions[0]) == 1:
            positions = [(x[0], 0) for x in positions]
        G = nx.from_numpy_array(conn.require("conn_mat"), create_using=nx.DiGraph())
        nx.draw_networkx_nodes(G, pos=positions, node_color="r", node_size=20)

        plt.show()
    if True:
        bp.visualize.raster_plot(t, X, title="Spikes of Excitatory Neurons", show=True)
    if True:
        R = X.sum(axis=0)
        R = R.reshape(FNSnet.E.size)
        plt.imshow(R, cmap="hot", interpolation="nearest")
        plt.show()
        print(R)
