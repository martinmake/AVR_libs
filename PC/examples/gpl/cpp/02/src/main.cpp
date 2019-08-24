#include <gpl/gpl.h>

#define CANVAS_WIDTH  800
#define CANVAS_HEIGHT 800
#define CANVAS_TITLE "TEST"

#define OUTER_PADDING 10
#define INNER_PADDING  2
#define BUTTON_SIZE   ((CANVAS_WIDTH - OUTER_PADDING * 2) / 28 - INNER_PADDING)

int main(void)
{
	using namespace Gpl;
	using namespace Gra::Math;

	Canvas canvas(CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_TITLE);
	Primitive::Container button_matrix(vec2<unsigned int>(OUTER_PADDING, OUTER_PADDING), vec2<unsigned int>(canvas.width() * 2 - OUTER_PADDING, canvas.height() - OUTER_PADDING));
	for (uint8_t y = 1; y <= 28; y++)
		for (uint8_t x = 1; x <= 28; x++)
			button_matrix << Primitive::Shape::Point(vec2<unsigned int>(x * (BUTTON_SIZE + INNER_PADDING), y * (BUTTON_SIZE + INNER_PADDING)), vec4<float>(0.5, 0.1, 1.0, 1.0), BUTTON_SIZE);

	canvas << button_matrix;
	canvas.animate();

	return 0;
}
