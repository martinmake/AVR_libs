#include <gpl/gpl.h>

#define CANVAS_WIDTH  800
#define CANVAS_HEIGHT 720
#define CANVAS_TITLE "TEST"

int main(void)
{
	using namespace Gpl;
	using namespace Gra::Math;

	Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_TITLE);
	for (uint8_t i = 0; i < 6; i++)
	{
		canvas << Primitive::Shape::Point(vec2<unsigned int>((i+1)*100,       (i+1)*100), vec4<float>(  i/5.0, 0.0, 1-i/5.0, 1.0), 80);
		canvas << Primitive::Shape::Point(vec2<unsigned int>((i+1)*100 + 100, (i+1)*100), vec4<float>(1-i/5.0, 0.0,   i/5.0, 1.0), 80);
	}

	canvas.animate();

	return 0;
}
