#include <gpl/gpl.h>

#define CANVAS_WIDTH  800
#define CANVAS_HEIGHT 800
#define CANVAS_TITLE "TEST"

#define OUTER_PADDING 100
#define INNER_PADDING  10
#define CONTAINER_WIDTH  (CANVAS_WIDTH  - OUTER_PADDING * 2)
#define CONTAINER_HEIGHT (CANVAS_HEIGHT - OUTER_PADDING * 2)

#define POINT_COLOR Color(0.3, 0.5, 0.8, 1.0)

int main(void)
{
	using namespace Gpl;
	using namespace Gra::Math;

	Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_TITLE);

	Primitive::Shape::Point point(Position(CONTAINER_WIDTH / 2, CONTAINER_HEIGHT / 2), POINT_COLOR, CONTAINER_WIDTH - INNER_PADDING * 2);
	point.on_mouse_over([](auto& event, void* point_instance)
	{
		Primitive::Shape::Point& point = *(Primitive::Shape::Point*) point_instance;
		point.color().w -= 0.001;
	});

	Primitive::Container container(Position(OUTER_PADDING, OUTER_PADDING), Size(CONTAINER_WIDTH, CONTAINER_HEIGHT));

	container << std::move(point    );
	canvas    << std::move(container);

	canvas.animate();

	return 0;
}
