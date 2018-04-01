import java.util.*;

class Example
{
	ArrayList<Double> attributes; // list of the attributes of the examples
	ArrayList<Integer> target; // target class; can take 0 or 1
    int number;

	/* Constructor */
	Example(ArrayList<Double> a, int t)
	{
		attributes = new ArrayList<Double>();
        number = t;
        target = new ArrayList<Integer>();
		for(double i : a)
        {
			attributes.add(i);
            for (int j = 0; j < t; j++)
                target.add(0);
            target.add(1);
            for (int j = t + 1; j < 10; j++)
		        target.add(0);
        }
	}
}
