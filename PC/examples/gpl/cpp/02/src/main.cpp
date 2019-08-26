#include <gpl/gpl.h>

#define CANVAS_WIDTH  880
#define CANVAS_HEIGHT 880
#define CANVAS_TITLE "TEST"

#define OUTER_PADDING 20
#define INNER_PADDING  2
#define BLOCK_SIZE    ((CANVAS_WIDTH - OUTER_PADDING * 2) / 28)
#define BUTTON_SIZE   (BLOCK_SIZE - INNER_PADDING * 2)
#define POINT_COLOR   Color(0.5, 0.1, 1.0, 1.0)

int main(void)
{
	using namespace Gpl;
	using namespace Gra::Math;

	Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_TITLE);
	Primitive::Container button_matrix(Position(OUTER_PADDING, OUTER_PADDING), Size(canvas.width() - OUTER_PADDING * 2 , canvas.height() - OUTER_PADDING * 2));
	for (uint8_t y = 0; y < 28; y++)
	for (uint8_t x = 0; x < 28; x++)
	{
		Primitive::Shape::Point point(Position(x * BLOCK_SIZE + BLOCK_SIZE / 2, y * BLOCK_SIZE + BLOCK_SIZE / 2), POINT_COLOR, BUTTON_SIZE);
	//	point.on_mouse_over([](auto& event)
	//	{
	//		std::cout << "[" << event.instance.position().x << ", " << event.instance.position().y << "] " << "ON MOUSE OVER" << std::endl;
	//		event.is_handled(true);
	//	});
		button_matrix << std::move(point);
	}

	canvas << button_matrix;
	canvas.animate();

	return 0;
}
