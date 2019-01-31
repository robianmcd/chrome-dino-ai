(function(){
    function getCanvas(nodeContainerElem) {
        let canvas = nodeContainerElem.querySelector('#layerCanvas1D');
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.id = 'layerCanvas1D';
            nodeContainerElem.appendChild(canvas);
        }

        return canvas;
    }

    const MAX_TRUNCATED_NODES = 35;

    window.LayerRenderer1DMixin = Vue.mixin({
        methods: {
            render1D: function(outputs, nodeContainerElem, truncate=true, normalize=true) {
                nodeWidth = 10;
                nodeHeight = 10;
                borderMargin = 2;
                containerWidth = nodeContainerElem.clientWidth - 20;
                nodesPerRow = Math.floor(containerWidth / nodeWidth) - 1;
                numRows = Math.ceil(outputs.length / nodesPerRow);

                let canvas = getCanvas(nodeContainerElem);
                let ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, canvas.width, canvas.height);

                canvas.width = (nodesPerRow + 1) * nodeWidth;
                canvas.height = truncate ? nodeHeight : numRows*nodeHeight;

                let radiusX = (nodeWidth - borderMargin) / 2;
                let radiusY = (nodeHeight - borderMargin) / 2;

                if(normalize) {
                    let minOutput = Math.min(...outputs);
                    outputs = outputs.map(o => o - minOutput);
                }
                let maxOutput = Math.max(...outputs);

                let numNodesShown = truncate ? Math.min(MAX_TRUNCATED_NODES, outputs.length) : outputs.length;

                for (let i = 0; i < numNodesShown; i++) {
                    output = outputs[i];

                    row = Math.floor(i / nodesPerRow);
                    iInRow = i % nodesPerRow;
                    rowXOffset = row % 2 === 1 ? nodeWidth/2 : 0;

                    let centerX = rowXOffset + iInRow * nodeWidth + nodeWidth / 2;
                    let centerY = row * nodeHeight + nodeHeight / 2;

                    if(truncate && i + 1 === MAX_TRUNCATED_NODES) {
                        ctx.font = "14px Arial";
                        ctx.fillStyle = "black";
                        ctx.fillText(`... ${outputs.length} ...`, centerX, centerY + 5);
                        centerX += 70;
                    }

                    outputPercent = Math.round(output/Math.max(1, maxOutput) * 100);

                    ctx.beginPath();
                    ctx.ellipse(centerX, centerY, radiusX, radiusY, 0, 0, 2 * Math.PI);

                    //ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI, false);
                    ctx.fillStyle = `hsl(200,80%,${100 - outputPercent/2}%)`;
                    ctx.fill();
                    ctx.lineWidth = 0.7;
                    ctx.strokeStyle = '#000000';
                    ctx.stroke();
                }
            }
        }
    });


})();
