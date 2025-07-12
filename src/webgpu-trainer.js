export class WebGPUTrainer {
    constructor(device) {
        this.device = device; this.buffers = {}; this.pipeline = null; this.modelSpec = null;
    }

    async init(spec) {
        this.modelSpec = spec;
        const { inputSize: H, nodesPerLayer: M, outputSize: K } = spec;

        this.buffers.weightsIH = this.device.createBuffer({
            size: H * M * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.buffers.weightsHO = this.device.createBuffer({
            size: M * K * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST
        });
        this.buffers.learningRate = this.device.createBuffer({
            size: 4, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.device.queue.writeBuffer(
            this.buffers.weightsIH, 0, Float32Array.from({ length: H * M }, () => Math.random() * .1 - .05));
        this.device.queue.writeBuffer(
            this.buffers.weightsHO, 0, Float32Array.from({ length: M * K }, () => Math.random() * .1 - .05));

        const code = /* wgsl */`
      struct S {input:array<f32,${H}>,label:array<f32,${K}>};
      @group(0) @binding(0) var<storage,read_write> wIH :array<f32>;
      @group(0) @binding(1) var<storage,read_write> wHO :array<f32>;
      @group(0) @binding(2) var<storage> SAMP :array<S>;
      @group(0) @binding(3) var<uniform> lr  :f32;
      fn sig(x:f32)->f32 {return 1.0/(1.0+exp(-x));}
      fn ds(y:f32)->f32  {return y*(1.0-y);}
      @compute @workgroup_size(1)
      fn main(@builtin(global_invocation_id) id:vec3<u32>){
        let s = SAMP[id.x];
        var h :array<f32,${M}>;
        for(var j=0u;j<${M}u;j++){
          var sm=0.0; for(var i=0u;i<${H}u;i++){sm+=s.input[i]*wIH[j*${H}u+i];}
          h[j]=sig(sm);
        }
        var o :array<f32,${K}>;
        for(var k=0u;k<${K}u;k++){
          var sm=0.0; for(var j=0u;j<${M}u;j++){sm+=h[j]*wHO[k*${M}u+j];}
          o[k]=sig(sm);
        }
        for(var k=0u;k<${K}u;k++){
          let gO=(o[k]-s.label[k])*ds(o[k]);
          for(var j=0u;j<${M}u;j++){
            let idxHO=k*${M}u+j;
            wHO[idxHO]-=lr*gO*h[j];
            let gH=gO*wHO[idxHO]*ds(h[j]);
            for(var i=0u;i<${H}u;i++){
              wIH[j*${H}u+i]-=lr*gH*s.input[i];
            }
          }
        }
      }`;
        const mod = this.device.createShaderModule({ code });
        this.pipeline = await this.device.createComputePipelineAsync({
            layout: 'auto', compute: { module: mod, entryPoint: 'main' }
        });
    }

    async train(samples,E,onEpoch){
        const { inputSize: H, outputSize: K } = this.modelSpec;
        const flat = []; samples.forEach(s => flat.push(...s.input, ...s.label));

        this.buffers.samples?.destroy?.();
        this.buffers.samples = this.device.createBuffer({
            size: flat.length * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });
        this.device.queue.writeBuffer(this.buffers.samples, 0, new Float32Array(flat));
        this.device.queue.writeBuffer(this.buffers.learningRate, 0, new Float32Array([0.1]));

        const bg = this.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this.buffers.weightsIH } },
                { binding: 1, resource: { buffer: this.buffers.weightsHO } },
                { binding: 2, resource: { buffer: this.buffers.samples } },
                { binding: 3, resource: { buffer: this.buffers.learningRate } }
            ]
        });
        for (let e = 0; e < E; ++e) {
            const enc = this.device.createCommandEncoder();
            const pass = enc.beginComputePass();
            pass.setPipeline(this.pipeline); pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(samples.length); pass.end();
            this.device.queue.submit([enc.finish()]);
            if(typeof onEpoch==='function') onEpoch(e+1,E);
        }
    }

    /* ---------------- save / load -------------------------------------- */
    async exportWeights() {
        const { inputSize: H, nodesPerLayer: M, outputSize: K } = this.modelSpec;
        const bytesIH = H * M * 4, bytesHO = M * K * 4;
        const readIH = this.device.createBuffer({ size: bytesIH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const readHO = this.device.createBuffer({ size: bytesHO, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.weightsIH, 0, readIH, 0, bytesIH);
        enc.copyBufferToBuffer(this.buffers.weightsHO, 0, readHO, 0, bytesHO);
        this.device.queue.submit([enc.finish()]);
        await readIH.mapAsync(GPUMapMode.READ); await readHO.mapAsync(GPUMapMode.READ);
        const wIH = Array.from(new Float32Array(readIH.getMappedRange()));
        const wHO = Array.from(new Float32Array(readHO.getMappedRange()));
        readIH.unmap(); readHO.unmap();
        return { meta: this.modelSpec, weightsIH: wIH, weightsHO: wHO };
    }

    async loadWeights(obj) {
        if (!obj.meta) throw 'weights file missing meta';
        const spec = obj.meta;

        /* exact match required */
        if (this.modelSpec &&
            (spec.inputSize !== this.modelSpec.inputSize ||
                spec.nodesPerLayer !== this.modelSpec.nodesPerLayer ||
                spec.outputSize !== this.modelSpec.outputSize)) {
            throw 'weights dimensions ≠ current trainer';
        }
        if (!this.modelSpec) await this.init(spec);

        const fIH = Float32Array.from(obj.weightsIH);
        const fHO = Float32Array.from(obj.weightsHO);
        this.device.queue.writeBuffer(this.buffers.weightsIH, 0, fIH);
        this.device.queue.writeBuffer(this.buffers.weightsHO, 0, fHO);
    }

    /* ---------------- CPU inference (returns K‑vector) ------------------ */
    async predictCPU(vec) {
        if (!this.modelSpec) throw new Error('model not initialised');
        const { inputSize: H, nodesPerLayer: M, outputSize: K } = this.modelSpec;
        vec = vec.slice(0, H).concat(Array(Math.max(0, H - vec.length)).fill(0));

        /* read weights back */
        const bytesIH = H * M * 4, bytesHO = M * K * 4;
        const rbIH = this.device.createBuffer({ size: bytesIH, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const rbHO = this.device.createBuffer({ size: bytesHO, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const enc = this.device.createCommandEncoder();
        enc.copyBufferToBuffer(this.buffers.weightsIH, 0, rbIH, 0, bytesIH);
        enc.copyBufferToBuffer(this.buffers.weightsHO, 0, rbHO, 0, bytesHO);
        this.device.queue.submit([enc.finish()]);
        await rbIH.mapAsync(GPUMapMode.READ); await rbHO.mapAsync(GPUMapMode.READ);
        const mapIH = new Float32Array(rbIH.getMappedRange());
        const mapHO = new Float32Array(rbHO.getMappedRange());
        const wIH = new Float32Array(mapIH.length);
        const wHO = new Float32Array(mapHO.length);
        wIH.set(mapIH); wHO.set(mapHO);
        rbIH.unmap(); rbHO.unmap();
        for (let i = 0; i < wIH.length; i++) if (!Number.isFinite(wIH[i])) wIH[i] = 0;
        for (let i = 0; i < wHO.length; i++) if (!Number.isFinite(wHO[i])) wHO[i] = 0;0.

        /* forward pass */
        const hid = new Float32Array(M);
        for (let j = 0; j < M; ++j) {
            let sum = 0; for (let i = 0; i < H; ++i) sum += vec[i] * wIH[j * H + i];
            hid[j] = 1 / (1 + Math.exp(-sum));
        }
        const out = new Float32Array(K);
        for (let k = 0; k < K; ++k) {
            let sum = 0; for (let j = 0; j < M; ++j) sum += hid[j] * wHO[k * M + j];
            out[k] = 1 / (1 + Math.exp(-sum));
            if (!Number.isFinite(out[k])) out[k] = 0;
        }
        return Array.from(out);
    }
}
