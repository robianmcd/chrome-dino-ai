(function () {
    let api = window.api;
    let modelStore = window.modelStore;
    let d3Controller = window.d3Controller;

    //Maps layer type to layer component name
    let layerCompMap = new Map();
    layerCompMap.set('InputLayer', 'input-layer');
    layerCompMap.set('Dense', 'dense-layer');
    layerCompMap.set('Flatten', 'flatten-layer');
    layerCompMap.set('Concatenate', 'concatenate-layer');
    layerCompMap.set('Conv2D', 'conv-2d-layer');

    let template = `
<div class="chrome-dino-app" ref="app">
    <template v-if="rowLayout">
        <div class="app__row" v-for="(row, rowI) in rowLayout" :ref="'row-' + rowI">
            <component 
                class="app__layer" v-for="layer in row" :key="layer.id" :id="layer.id" :ref="'layer-' + layer.id"
                :is="getLayerCompName(layer)" :layer="layer" :layerOutput="layerOutputs && layerOutputs[layer.id]">
            </component>
        </div>
    </template>
    <canvas id="app__edge-canvas" class="app__edge-canvas" ref="edgeCanvas"></canvas>
</div>
`;

    Vue.component('chrome-dino-app', {
        template,
        data: () => ({
            model: undefined,
            rowLayout: undefined,
            layerOutputs: undefined
        }),
        created: function () {
            modelStore.load_model()
                .then(() => {
                    this.rowLayout = modelStore.rowLayout;
                    setTimeout(() => {
                        d3Controller.applyToContainer('.chrome-dino-app', '.app__layer');
                    });
                })
                .then(() => this.getNextOutput());
        },
        methods: {
            getLayerCompName: function (layer) {
                if (layerCompMap.has(layer['class_name'])) {
                    return layerCompMap.get(layer['class_name']);
                } else {
                    return 'generic-layer'
                }
            },

            getNextOutput() {
                return api.getLayerOutputs()
                    .then((layerOutputs => {
                        this.layerOutputs = layerOutputs;
                        setTimeout(() => this.getNextOutput(), 125);
                        setTimeout(() => {
                            this.resizeRows();

                            modelStore.model.layers.forEach(layer => {
                                layer.comp && layer.comp.drawEdges && layer.comp.drawEdges();
                            });

                        });
                    }));
            },

            resizeRows: function() {
                let maxRowWidth = 0;
                modelStore.rowLayout.forEach((row, rowI) => {
                    let rowHeight = row
                        .map(layer => this.$refs['layer-' + layer.id][0].$el)
                        .reduce((maxHeight, layerElem) => Math.max(layerElem.clientHeight, maxHeight), 0);

                    let rowMargin = 10;
                    let rowWidth = rowMargin*2 + row
                        .map(layer => {
                            let layerElem = this.$refs['layer-' + layer.id][0].$el;
                            return layerElem.offsetLeft + layerElem.offsetWidth;
                        })
                        .reduce((maxRight, layerRight) => Math.max(layerRight, maxRight), 0);
                    maxRowWidth = Math.max(rowWidth, maxRowWidth);

                    this.$refs['row-' + rowI][0].style.height = rowHeight + 'px';
                });


                this.$refs['app'].style.width = maxRowWidth + 'px';
                let canvas = this.$refs['edgeCanvas'];
                canvas.width = canvas.clientWidth;
                canvas.height = canvas.clientHeight;

            }
        }

    });
})();
