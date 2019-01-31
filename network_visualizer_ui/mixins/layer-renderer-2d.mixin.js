(function(){
    pointSize = 3;
    pointArea = pointSize**2;

    function getCanvasByName(nodeContainerElem, name) {
        let canvas = nodeContainerElem.querySelector('#' + name);
        if (!canvas) {
            canvas = document.createElement('canvas');
            canvas.id = name;
            nodeContainerElem.appendChild(canvas);
        }

        return canvas;
    }

    function imageBuffersFromChannelArray(channelArray) {
        height = channelArray.length;
        width = channelArray[0].length;
        numChannels = channelArray[0][0].length;

        imgBuffers = [];

        for(let channel = 0; channel < numChannels; channel++) {
            buffer = new Uint8ClampedArray(width * height * pointArea * 4);

            //Based on https://stackoverflow.com/questions/22823752/creating-image-from-array-in-javascript-and-html5
            for(let x = 0; x < width; x++) {
                for(let y = 0; y < height; y++) {

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

    window.LayerRenderer2DMixin = Vue.mixin({
        methods: {
            render2D: function (outputs, nodeContainerElem) {
                let buffers = imageBuffersFromChannelArray(outputs);

                let height = outputs.length;
                let width = outputs[0].length;
                let channels = outputs[0][0].length;

                let maxChannels = 300;
                channels = Math.min(channels, maxChannels);

                for(let channel = 0; channel < channels; channel++) {
                    // create off-screen canvas element
                    let canvas = getCanvasByName(nodeContainerElem, 'canvas-' + channel);
                    canvas.style = 'border: 1px solid black';
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
                }
            }
        }
    });

})();
