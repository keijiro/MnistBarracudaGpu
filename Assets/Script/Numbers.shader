Shader "MnistBarracuda/Label"
{
    Properties
    {
        _MainTex("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Pass
        {
            Blend One One

            CGPROGRAM

            #pragma vertex Vertex
            #pragma fragment Fragment

            #include "UnityCG.cginc"

            sampler2D _MainTex;
            StructuredBuffer<float> _Scores;

            void Vertex(float4 position : POSITION,
                        float2 uv : TEXCOORD0,
                        out float4 outPosition : SV_Position,
                        out float2 outUV : TEXCOORD0)
            {
                outPosition = UnityObjectToClipPos(position);
                outUV = uv;
            }

            float4 Fragment(float4 position : SV_Position,
                           float2 uv : TEXCOORD) : SV_Target
            {
                float alpha = tex2D(_MainTex, uv).a;
                float score = _Scores[(uint)(uv.x * 10)];
                return lerp(0.1, 1, score) * alpha;
            }

            ENDCG
        }
    }
}
