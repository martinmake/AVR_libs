#version 330

#define PI 3.14159265358979323846

in vec4 gl_FragCoord;
in vec2 gl_PointCoord;

out vec4 color;

uniform vec2 u_resolution;
uniform float u_time;

float rand(vec2 c);
float noise(vec2 p, float freq);
float perlin_noise(vec2 p, int res);

void main()
{
	vec2 pixel = gl_FragCoord.xy / u_resolution.xy;
	float t = u_time;
	vec4 background_color = vec4(0.0, 0.0, 0.0, 0.0);

	float r = pixel.x;
	float g = 0.6;
	float b = 1 - pixel.x;
	float a = 1.0;

	r *= sin(t +   0);
	g *= sin(t +  90);
	b *= sin(t + 180);

	if (length(pixel - vec2(0.5, 0.5)) > 0.5)
		color = background_color;
	else
	{
		a = noise(pixel, 1000);
		color = vec4(r, g, b, a);
	}
}

float rand(vec2 c)
{
	return fract(sin(dot(c.xy ,vec2(12.9898,78.233))) * 43758.5453);
}

float noise(vec2 p, float freq)
{
	float unit = u_resolution.x/freq;
	vec2 ij = floor(p/unit);
	vec2 xy = mod(p,unit)/unit;
	//xy = 3.*xy*xy-2.*xy*xy*xy;
	xy = .5*(1.-cos(PI*xy));
	float a = rand((ij+vec2(0.,0.)));
	float b = rand((ij+vec2(1.,0.)));
	float c = rand((ij+vec2(0.,1.)));
	float d = rand((ij+vec2(1.,1.)));
	float x1 = mix(a, b, xy.x);
	float x2 = mix(c, d, xy.x);
	return mix(x1, x2, xy.y);
}

float perlin_noise(vec2 p, int res)
{
	float persistance = .5;
	float n = 0.;
	float normK = 0.;
	float f = 4.;
	float amp = 1.;
	int iCount = 0;
	for (int i = 0; i<50; i++)
	{
		n+=amp*noise(p, f);
		f*=2.;
		normK+=amp;
		amp*=persistance;
		if (iCount == res) break;
		iCount++;
	}
	float nf = n/normK;
	return nf*nf*nf*nf;
}
