#include "widgets/text_box.h"

TextBox::TextBox(int x, int y, const std::vector<std::string>& text)
	: Widget(x, y, max_string(text), text.size())
{
}

TextBox::~TextBox()
{
}
