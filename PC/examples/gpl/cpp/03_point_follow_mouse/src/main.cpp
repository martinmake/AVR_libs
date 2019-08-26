#include <gpl/gpl.h>

#define CANVAS_WIDTH  800
#define CANVAS_HEIGHT 800
#define CANVAS_TITLE "TEST"

#define POINT_SIZE 10

#define POINT_COLOR Color(0.3, 0.5, 0.8, 1.0)

int main(void)
{
	using namespace Gpl;
	using namespace Gra::Math;

	Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_TITLE);

	Primitive::Shape::Point point(Position(POINT_SIZE, POINT_SIZE), POINT_COLOR, POINT_SIZE);
	canvas << std::move(point);

	canvas.animate();

	return 0;
}
