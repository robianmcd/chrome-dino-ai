let template = `
<div class="conv-2d-layer">
        <div class="layer__info">
        <div class="layer__title">Conv 2D</div>
        <div class="layer__info-row">
            <div class="layer__info-label">Activation:</div>
            <div class="layer__info-value">{{layer.config.activation}}</div>
        </div>
        <div class="layer__info-row">
            <div class="layer__info-label">Kernal:</div>
            <div class="layer__info-value">{{layer.config.kernel_size}}</div>
        </div>
        <div class="layer__info-row">
            <div class="layer__info-label">Stride:</div>
            <div class="layer__info-value">{{layer.config.strides}}</div>
        </div>
        <div class="layer__info-row">
            <div class="layer__info-label">Output Shape:</div>
            <div class="layer__info-value">{{layer.outputShape.join('x')}}</div>
        </div>
    </div>
    <div class="layer__nodes" ref="node_container"></div>
</div>
`;

Vue.component('conv-2d-layer', {
    template,
    mixins: [window.LayerRenderer2DMixin],
    props: ['layer', 'layerOutput'],
    methods: {

    },
    watch: {
        layerOutput: function() {
            this.render2D(this.layerOutput, this.$refs['node_container']);
        }
    }
});
