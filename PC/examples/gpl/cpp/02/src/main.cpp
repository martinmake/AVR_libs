#include <gpl/gpl.h>

#define CANVAS_WIDTH  880
#define CANVAS_HEIGHT 880
#define CANVAS_TITLE "TEST"

#define OUTER_PADDING 20
#define INNER_PADDING  2
#define BLOCK_SIZE    ((CANVAS_WIDTH - OUTER_PADDING * 2) / 28)
#define BUTTON_SIZE   (BLOCK_SIZE - INNER_PADDING * 2)

#define POINT_COLOR_BUTTON_PRESSED_LEFT  Color(0.3, 0.6, 0.9, 1.0)
#define POINT_COLOR_BUTTON_PRESSED_RIGHT Color(0.5, 0.1, 1.0, 1.0)
#define POINT_COLOR_INITIAL              POINT_COLOR_BUTTON_PRESSED_RIGHT

int main(void)
{
	using namespace Gpl;
	using namespace Gra;
	using namespace Gra::Input::Window;
	using namespace Gra::Math;

	Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_TITLE);
	Primitive::Container button_matrix(Position(OUTER_PADDING, OUTER_PADDING), Size(canvas.width() - OUTER_PADDING * 2 , canvas.height() - OUTER_PADDING * 2));
	for (uint8_t y = 0; y < 28; y++)
	for (uint8_t x = 0; x < 28; x++)
	{
		Primitive::Shape::Point point(Position(x * BLOCK_SIZE + BLOCK_SIZE / 2, y * BLOCK_SIZE + BLOCK_SIZE / 2), POINT_COLOR_INITIAL, BUTTON_SIZE);
		point.on_mouse_over([&](auto& event, void* instance)
		{
			Primitive::Shape::Point& point = *(Primitive::Shape::Point*) instance;
			if      (canvas.mouse_button(Mouse::Button::LEFT) == Mouse::Action::PRESS)
				point.color(POINT_COLOR_BUTTON_PRESSED_LEFT);
			else if (canvas.mouse_button(Mouse::Button::RIGHT) == Mouse::Action::PRESS)
				point.color(POINT_COLOR_BUTTON_PRESSED_RIGHT);
		});
		button_matrix << std::move(point);
	}

	canvas << button_matrix;
	canvas.animate();

	return 0;
}
