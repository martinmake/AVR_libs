#include <gpl/canvas.h>
#include <gpl/primitives/all.h>
#include <gpl/primitive.h>

#define WINDOW_WIDTH  800
#define WINDOW_HEIGHT 720
#define WINDOW_NAME  "TEST"

int main(void)
{
	using namespace Gpl;
	using namespace Gra;

	Gpl::Canvas::s_renderer.init(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);

	Canvas canvas;
	for (uint8_t i = 0; i < 6; i++)
	{
		canvas << new Primitive::Point(Math::vec3<float>((i+1)*100,       (i+1)*100, 0), Math::vec4<float>(  i/5.0, 0.0, 1-i/5.0, 1.0), 80);
		canvas << new Primitive::Point(Math::vec3<float>((i+1)*100 + 100, (i+1)*100, 0), Math::vec4<float>(1-i/5.0, 0.0,   i/5.0, 1.0), 80);
	}

	while (!Canvas::s_renderer.should_close())
		canvas.render();

	return 0;
}
