# Generated by Django 3.2.9 on 2021-11-26 02:37

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Music_streamer', '0003_alter_usersong_url'),
    ]

    operations = [
        migrations.AlterField(
            model_name='usersong',
            name='url',
            field=models.CharField(max_length=400),
        ),
    ]
