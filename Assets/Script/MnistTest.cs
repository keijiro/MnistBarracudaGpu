using UnityEngine;
using Unity.Barracuda;

sealed class MnistTest : MonoBehaviour
{
    public NNModel _model;
    public ComputeShader _preprocess;
    public ComputeShader _postprocess;
    public Texture2D _sourceImage;
    public Renderer _previewRenderer;
    public Renderer _labelRenderer;

    ComputeBuffer _scores;

    void Start()
    {
        // Prepare the input tensor buffer.
        var shape = new TensorShape(1, 28, 28, 1);
        var data = new ComputeTensorData
          (shape, "Input", ComputeInfo.ChannelsOrder.NHWC);
        using var input = new Tensor(shape, data);

        // Invoke the preprocessing compute kernel.
        _preprocess.SetTexture(0, "Input", _sourceImage);
        _preprocess.SetBuffer(0, "Output", data.buffer);
        _preprocess.Dispatch(0, 28 / 4, 28 / 4, 1);

        // Run the MNIST model.
        using var worker =
          ModelLoader.Load(_model).CreateWorker(WorkerFactory.Device.GPU);

        worker.Execute(input);

        // Retrieve the results into a temporary render texture.
        var rt = RenderTexture.GetTemporary(10, 1, 0, RenderTextureFormat.RFloat);
        using (var tensor = worker.PeekOutput().Reshape(new TensorShape(1, 1, 10, 1)))
            tensor.ToRenderTexture(rt);

        // Invoke the postprocessing compute kernel.
        _scores = new ComputeBuffer(10, sizeof(float));
        _postprocess.SetTexture(0, "Input", rt);
        _postprocess.SetBuffer(0, "Output", _scores);
        _postprocess.Dispatch(0, 1, 1, 1);

        RenderTexture.ReleaseTemporary(rt);

        // Output display
        _previewRenderer.material.mainTexture = _sourceImage;
        _labelRenderer.material.SetBuffer("_Scores", _scores);
    }

    void OnDestroy()
      => _scores?.Dispose();
}
