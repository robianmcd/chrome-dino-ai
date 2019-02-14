(function(){
    pointSize = 3;
    pointArea = pointSize**2;

    function getCanvasByName(nodeContainerElem, name) {
        let canvas = nodeContainerElem.querySelector('#' + name);
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.id = name;
            canvas.className = 'layer__canvas-2d';
            nodeContainerElem.appendChild(canvas);
        }

        return canvas;
    }

    function imageBuffersFromChannelArray(channelArray, pixelPositions) {
        height = channelArray.length;
        width = channelArray[0].length;
        numChannels = channelArray[0][0].length;

        imgBuffers = [];

        for(let channel = 0; channel < numChannels; channel++) {
            buffer = new Uint8ClampedArray(width * height * pointArea * 4);

            //Based on https://stackoverflow.com/questions/22823752/creating-image-from-array-in-javascript-and-html5
            for(let y = 0; y < height; y++) {
                for(let x = 0; x < width; x++) {
                    // let pixelIndex = channel + x * numChannels + y * numChannels * width;
                    // pixelPositions[pixelIndex] = {x: x*pointSize, y: y*pointSize, channel: channel};
                    pixelPositions.push({x: x*pointSize, y: y*pointSize, channel: channel});

                    for (let innerPointXOffset = 0; innerPointXOffset < pointSize; innerPointXOffset++) {
                        for (let innerPointYOffset = 0; innerPointYOffset < pointSize; innerPointYOffset++) {
                            pixelRow = y * pointSize + innerPointYOffset;
                            pixelCol = x * pointSize + innerPointXOffset;
                            pixelOffset = (pixelRow * width * pointSize + pixelCol) * 4;
                            greyscaleValue = channelArray[y][x][channel] * 255;

                            buffer[pixelOffset] = greyscaleValue; // Red
                            buffer[pixelOffset + 1] = greyscaleValue; // Green
                            buffer[pixelOffset + 2] = greyscaleValue; // Blue
                            buffer[pixelOffset + 3] = 255; // Alpha
                        }
                    }

                }
            }

            imgBuffers.push(buffer)
        }

        return imgBuffers
    }

    window.LayerRenderer2DMixin = {
        data: function() {
            return {
                layerRenderer2DPixelPositions: [],
                layerRenderer2DNodeElems: []
            };
        },
        methods: {
            render2D: function (outputs, nodeContainerElem) {
                this.layerRenderer2DNodeElems = [];
                this.layerRenderer2DPixelPositions = [];

                let buffers = imageBuffersFromChannelArray(outputs, this.layerRenderer2DPixelPositions);

                let height = outputs.length;
                let width = outputs[0].length;
                let channels = outputs[0][0].length;

                let maxChannels = 7;
                channels = Math.min(channels, maxChannels);

                for(let channel = 0; channel < channels; channel++) {
                    // create off-screen canvas element
                    let canvas = getCanvasByName(nodeContainerElem, 'canvas-' + channel);
                    canvas.style = 'border: 1px solid #838383';
                    let ctx = canvas.getContext('2d');
                    ctx.clearRect(0, 0, canvas.width, canvas.height);

                    canvas.width = width * pointSize;
                    canvas.height = height * pointSize;

                    // create imageData object
                    var imageData = ctx.createImageData(width * pointSize, height * pointSize);

                    // set our buffer as source
                    imageData.data.set(buffers[channel]);

                    // update canvas with new data
                    ctx.imageSmoothingEnabled = false;
                    ctx.putImageData(imageData, 0, 0);

                    this.layerRenderer2DNodeElems.push(canvas);

                }
            },
            get2DNodePositions: function() {
                return this.layerRenderer2DNodeElems.map(nodeCanvas => {
                    let canvasRect = nodeCanvas.getBoundingClientRect();
                    return {
                        x: canvasRect.left + document.documentElement.scrollLeft + canvasRect.width/2,
                        y: canvasRect.top + document.documentElement.scrollTop + canvasRect.height/2
                    };
                });

            },
            get1DNodePositions: function() {
                return this.layerRenderer2DPixelPositions
                    .filter(pixelPos => pixelPos.channel < this.layerRenderer2DNodeElems.length)
                    .map(pixelPos => {
                        let canvasRect = this.layerRenderer2DNodeElems[pixelPos.channel].getBoundingClientRect();
                        return {
                            x: canvasRect.left + pixelPos.x + document.documentElement.scrollLeft + pointSize/2,
                            y: canvasRect.top + pixelPos.y + document.documentElement.scrollTop + pointSize/2
                        };
                    });
            }
        }
    };

})();
