{
using namespace Cabs::Position;

Widget& sample_widget = *new Widget(CENTER, CENTER, 21, 10);
sample_widget.label("SAMPLE WIDGET");
sample_widget.box   (true);
sample_widget.shadow(true);

screen << sample_widget;
}
