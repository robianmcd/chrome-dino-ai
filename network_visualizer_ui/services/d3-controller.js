(function() {
    let modelStore = window.modelStore;

    class D3Controller {
        constructor() {
            this.nodeWidth = 670;
        }

        applyToContainer(containerSelector, layerSelector) {
            if(!modelStore.initialized) {
                throw new Error('modelStore should be initialized before applying the d3 layout');
            }

            modelStore.model.layers.forEach(layer => {
                layer.x = 0;
                layer.vx = 0;
            });

            let nodes = modelStore.model.layers;

            let d3ContainerSelector = d3.select(containerSelector);
            let simulation = d3.forceSimulation(nodes)
                .force('customForce', this._getCustomForce(nodes));

            //For some reason creating the simulation messes with each node's x value even if there aren't any forces so
            //this resets them all.
            nodes.forEach(n => n.x = 0);

            let d3NodeSelector = d3ContainerSelector.selectAll(layerSelector)
                .data(nodes, function(d) {return d ? d.id : this.id})
                .style("left", function(d) { return d.x + "px"; });

            simulation.on("tick", () => {
                //Align leftmost node with the left side of the container and move all other notes the same amount.
                let minX = nodes.reduce((min, node) => Math.min(min, node.x), nodes[0].x);
                nodes.forEach(node => node.x += minX * -1);

                d3NodeSelector.style("left", d => d.x + 'px');
            });

            // See https://github.com/d3/d3-force/blob/master/README.md#simulation_tick
            for (var i = 0, n = Math.ceil(Math.log(simulation.alphaMin()) / Math.log(1 - simulation.alphaDecay())); i < n; ++i) {
                simulation.tick();
            }
        }

        //Examples of this force:
        // https://codepen.io/robianmcd/pen/VgjryL
        // https://codepen.io/robianmcd/pen/ErybXr
        // https://codepen.io/robianmcd/pen/xMVyNm
        // https://codepen.io/robianmcd/pen/qgZJmy
        _getCustomForce(nodes) {
            return (alpha) => {
                nodes.forEach((node, i) => {
                    let linkedNodes = [...node.inboundLayerNames, ...node.outboundLayerNames]
                        .map(name => modelStore.layerMap.get(name));


                    let totalSquaredOffset = linkedNodes
                        .reduce((offsetAgg, linkedNode) => {
                            let horizDist = linkedNode.x - node.x;

                            return offsetAgg + horizDist * Math.abs(horizDist);
                        }, 0);

                    let directionMult = totalSquaredOffset > 0 ? 1 : -1;
                    node.vx += Math.sqrt(Math.abs(totalSquaredOffset)) * directionMult * alpha / 2;

                    nodes
                        .filter(n => n.row === node.row && n.id !== node.id)
                        .forEach((sameRowNode) => {
                            if (sameRowNode.x === node.x) {
                                let directionMult = (i < sameRowNode.index) ? -1 : 1;
                                node.vx = this.nodeWidth / 2 * directionMult * alpha;
                            } else if(sameRowNode.x > node.x && sameRowNode.x <= node.x + this.nodeWidth ||
                                node.x > sameRowNode.x && node.x <= sameRowNode.x + this.nodeWidth)
                            {
                                let directionMult = node.x > sameRowNode.x ? 1 : -1;
                                node.vx += (this.nodeWidth - Math.abs(node.x - sameRowNode.x)) * directionMult * Math.sqrt(alpha);
                            }
                        });

                });

            };
        }
    }

    window.d3Controller = new D3Controller();
})();

