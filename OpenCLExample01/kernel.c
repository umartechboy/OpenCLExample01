kernel void GetIfPrime(global int* message)
{
    int index = get_global_id(0);

    int upperl = (int)sqrt((float)message[index]);
    for (int i = 2; i <= upperl; i++)
    {
        if (message[index] % i == 0)
        {
            //printf("" %d / %d\n"",index,i );
            message[index] = 0;
            return;
        }
    }
    //printf("" % d"",index);
}