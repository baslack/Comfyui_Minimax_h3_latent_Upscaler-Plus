import { app } from "../../../scripts/app.js";

// When a MinimaxH3LatentUpscaler3DProvider is wired into the optional
// "learned_upscaler" input, the node ignores its own model_name/device/
// precision/offload_after_upscale widgets (see execute() in
// minimax_h3_latent_upscaler_3d.py). Grey those widgets out while that
// input is connected so it's not ambiguous which one is actually in
// effect, and restore them on disconnect.

const NODE_NAME = "MinimaxH3LatentUpscaler3D";
const CONTROLLED_WIDGETS = ["model_name", "device", "precision", "offload_after_upscale"];
const PROVIDER_INPUT = "learned_upscaler";

function setControlledWidgetsDisabled(node, disabled) {
    let changed = false;
    for (const name of CONTROLLED_WIDGETS) {
        const widget = node.widgets?.find((w) => w.name === name);
        if (widget && widget.disabled !== disabled) {
            widget.disabled = disabled;
            changed = true;
        }
    }
    if (changed) {
        node.graph?.setDirtyCanvas(true, true);
    }
}

function isProviderConnected(node) {
    const slot = node.inputs?.find((inp) => inp.name === PROVIDER_INPUT);
    return !!(slot && slot.link != null);
}

app.registerExtension({
    name: "comfyui-minimax-h3-latent-upscaler.upscaler3d_widgets",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) {
            return;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, linkInfo, ioSlot) {
            const result = onConnectionsChange?.apply(this, arguments);
            const INPUT = 1; // LiteGraph.INPUT; avoid depending on the global being in scope
            if (type !== INPUT) {
                return result;
            }
            const slot = this.inputs?.[index];
            if (!slot || slot.name !== PROVIDER_INPUT) {
                return result;
            }
            setControlledWidgetsDisabled(this, connected);
            return result;
        };

        // Saved workflows restore links while the node is being configured, before
        // (or in a different order than) onConnectionsChange settles reliably, so
        // re-check once the node has finished loading.
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = onConfigure?.apply(this, arguments);
            setTimeout(() => setControlledWidgetsDisabled(this, isProviderConnected(this)), 100);
            return result;
        };

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            setTimeout(() => setControlledWidgetsDisabled(this, isProviderConnected(this)), 100);
            return result;
        };
    },
});
