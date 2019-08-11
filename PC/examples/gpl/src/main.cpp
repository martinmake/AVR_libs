#include <gpl/canvas.h>
#include <gpl/primitives/all.h>
#include <gpl/primitive.h>

#define WINDOW_WIDTH  200
#define WINDOW_HEIGHT 200
#define WINDOW_NAME   "TEST"

Gra::Renderer Gpl::Canvas::s_renderer(WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_NAME);

int main(void)
{
	using namespace Gpl;

	Canvas canvas;
	canvas << new Primitive::Point({100, 100, 0}, {0.7, 0.2, 0.5, 1.0}, 100);

	while (!Canvas::s_renderer.should_close())
		canvas.render();

	return 0;
}
