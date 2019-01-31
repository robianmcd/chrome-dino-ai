(function(){

    let template = `
<div class="input-layer">
    <div class="layer__info">
        <div class="layer__title">Flatten</div>
        <div class="layer__info-row">
            <div class="layer__info-label">Output Shape:</div>
            <div class="layer__info-value">{{layer.outputShape.join('x')}}</div>
        </div>
    </div>
    <div class="layer__nodes" ref="node_container"></div>
</div>
`;

    Vue.component('flatten-layer', {
        template,
        mixins: [window.LayerRenderer1DMixin],
        props: ['layer', 'layerOutput'],
        methods: {

        },
        watch: {
            layerOutput: function() {
                this.render1D(this.layerOutput, this.$refs['node_container']);
            }
        }
    });

})();
