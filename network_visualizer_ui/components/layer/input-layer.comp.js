(function(){

    let template = `
<div class="input-layer">
    <div class="layer__info">
        <div class="layer__title">Input</div>
        <div class="layer__info-row">
            <div class="layer__info-label">Output Shape:</div>
            <div class="layer__info-value">{{layer.outputShape.join('x')}}</div>
        </div>
    </div>
    <div class="layer__nodes" ref="node_container"></div>
</div>
`;

    Vue.component('input-layer', {
        template,
        mixins: [window.LayerRenderer1DMixin, window.LayerRenderer2DMixin],
        props: ['layer', 'layerOutput'],
        methods: {

        },
        watch: {
            layerOutput: function() {
                if(this.layer.outputShape.length === 1) {
                    this.render1D(this.layerOutput, this.$refs['node_container']);
                } else if (this.layer.outputShape.length === 3) {
                    this.render2D(this.layerOutput, this.$refs['node_container']);
                }

            }
        }
    });

})();
